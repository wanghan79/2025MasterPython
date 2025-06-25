import numpy as np
import random
from tqdm import tqdm, trange
import os
# Specify CUDA_VISIBLE_DEVICES in the command,
# e.g., CUDA_VISIBLE_DEVICES=0,1 nohup bash exp_on_b7server_0.sh
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import time
import json
import warnings

warnings.filterwarnings('ignore')
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from collections import OrderedDict
from torch.cuda.amp import GradScaler, autocast

from utils.parser_utils import get_args
from utils.logger_utils import get_logger
from utils.other_utils import *
from utils.optimization_utils import *
from utils.mixout_utils import *
from modeling.bert_models import *

def generate_refs(self, model=None, load_cache=True):
    input_data = self.input_data
    sim_thres = self.args.CET_sim_thres  # 相似度阈值，用于筛选参考样本

    # 构建缓存路径，用于避免重复计算
    cache_path = os.path.join(
        self.args.dataset_dir,
        'ref_str_{}_{}_nsamples{}_top{}_{}.pk'.format(
            self.args.input_format,
            self.args.pretrain_model,
            len(input_data['input_str']),
            self.args.CET_topk,
            'thres%.2f' % sim_thres
        )
    )  # 示例路径：data/obqa/official/ref_str_each_option_roberta-large_nsamples4957_top5_thres1.00.pk

    # 如果启用缓存且缓存文件存在，则直接加载缓存内容
    if load_cache and os.path.isfile(cache_path):
        logger.info('Loading cache for ref str from %s' % cache_path)
        with open(cache_path, 'rb') as f:
            cache_dict = pickle.load(f)
            input_data['ref_str'] = cache_dict.get('ref_str')
            input_data['ref_cnt'] = cache_dict.get('ref_cnt')
        return input_data

    # 加载 spaCy 的大型英语模型，用于文本相似度计算
    nlp = spacy.load('en_core_web_lg')

    # 提取每个样本的正确答案文本，转为小写去除空白
    gt_answer_lst = [eds[i].strip().lower() for eds, i in zip(input_data['endings'], input_data['example_label'])]

    n_samples = len(gt_answer_lst)  # 样本总数，如 4957
    sim_matrix = np.zeros((n_samples, n_samples))  # 初始化相似度矩阵

    # 用 spaCy 编码所有答案文本，得到 doc 对象
    doc_lst = [nlp(ans) for ans in gt_answer_lst]

    # 构建对称的相似度矩阵（只计算上三角）
    for i in range(n_samples):
        for j in range(n_samples):
            if i < j:
                continue
            sim_score = doc_lst[i].similarity(doc_lst[j])
            sim_matrix[i][j] = sim_score
            sim_matrix[j][i] = sim_score  # 相似度矩阵是对称的

    # 将自身与自身的相似度设为极小值（防止被选中为参考样本）
    sim_matrix = sim_matrix - np.eye(n_samples) * 1e8

    # 从相似度矩阵中找出每个样本最相似的 topk 个参考样本（含索引和得分）
    match_sim_matrix, match_id_matrix = torch.topk(
        torch.from_numpy(sim_matrix),
        k=self.args.CET_topk,
        largest=True,
        dim=1
    )  # 输出大小为 (n_samples, topk)

    # 初始化存储所有参考样本内容和参考数量
    ref_str_all = []
    ref_cnt_all = []

    # 遍历所有样本，生成参考字符串
    for i in range(n_samples):
        if self.args.input_format == 'each_option':
            # 每个选项单独构建参考内容
            ref_str_lst = []
            n_option = len(input_data['input_str'][i])
            ref_cnt = 0
            for option_id in range(n_option):
                option_str = input_data['endings'][i][option_id]  # 当前选项文本
                for k in range(self.args.CET_topk):
                    match_sim = match_sim_matrix[i][k]

                    # 不符合阈值或非法相似度，则用自身作为参考
                    if sim_thres > 0 and match_sim < sim_thres:
                        match_id = i
                    elif match_sim > 1.0:
                        match_id = i
                    else:
                        match_id = match_id_matrix[i][k]
                        if option_id == 0:
                            ref_cnt += 1

                    # 获取匹配参考样本的上下文文本并拼接选项，加入参考列表
                    one_ref_question = input_data['contexts'][match_id][0]
                    ref_str_lst.append(one_ref_question + ' ' + option_str)
            ref_str_all.append(ref_str_lst)
            ref_cnt_all.append(ref_cnt)

        elif self.args.input_format == 'all_option':
            # 所有选项合并为一个完整字符串进行参考匹配
            ref_str_lst = []
            n_option = len(input_data['endings'][i])
            ref_cnt = 0
            option_str = ' \\n '
            for ed_idx, ed in enumerate(input_data['endings'][i]):
                option_str += '(' + chr(ord('A') + ed_idx) + ')' + ' ' + ed + ' '

            for k in range(self.args.CET_topk):
                match_sim = match_sim_matrix[i][k]
                if sim_thres > 0 and match_sim < sim_thres:
                    match_id = i
                elif match_sim > 1.0:
                    match_id = i
                else:
                    match_id = match_id_matrix[i][k]
                    ref_cnt += 1

                one_ref_question = input_data['contexts'][match_id]
                ref_str_lst.append(one_ref_question + ' ' + option_str)

            ref_str_all.append(ref_str_lst)
            ref_cnt_all.append(ref_cnt)

        else:
            raise Exception('Invalid input_format %s' % (self.args.input_format))

    # 将参考样本字符串和参考计数添加到输入数据中
    input_data['ref_str'] = ref_str_all
    input_data['ref_cnt'] = ref_cnt_all

    # 保存缓存以便下次使用
    with open(cache_path, 'wb') as f:
        logger.info('Saving cache for ref str to %s' % (cache_path))
        pickle.dump({
            'ref_str': ref_str_all,
            'ref_cnt': ref_cnt_all,
            'match_sim_matrix': match_sim_matrix,
            'match_id_matrix': match_id_matrix,
        }, f)

    # 更新数据
    self.input_data = input_data

def compute_CET_loss(self, input_data, labels):
    # 当前 batch 的大小（样本数）
    bs = len(input_data['example_id'])

    # 每个样本对应的选项个数
    nc = input_data['LM_input']['input_ids'].shape[0] // bs

    # 得到模型对每个输入的输出 logits，并 reshape 成 (bs, nc)
    logits = self.forward(input_data['LM_input']).reshape(bs, nc)
    assert logits.shape == (len(labels), nc)  # 断言维度正确

    # 对 logits 应用 softmax 得到归一化概率
    prob_score = torch.softmax(logits, dim=-1)  # 形状：(bs, nc)

    # 统计整个 batch 内所有样本的参考数量之和
    batch_ref_cnt = np.sum(input_data['ref_cnt']).item()

    # 如果该 batch 中所有样本都没有参考样本，则直接使用原始概率
    if batch_ref_cnt == 0:
        joint_prob_score = prob_score
    else:
        # 否则处理所有参考输入，期望形状为 (nc * batch_ref_cnt, seq_len)
        assert input_data['ref_LM_input']['input_ids'].shape[0] == nc * batch_ref_cnt

        # 将参考样本按 batch 切分计算 logits，避免一次性前向传播爆显存
        num_chunk = (batch_ref_cnt - 1) // self.args.batch_size + 1  # 根据参考数决定切块数量
        ref_logits_lst = []
        for chunk_input_ids, chunk_attention_mask in zip(
            input_data['ref_LM_input']['input_ids'].chunk(num_chunk, 0),
            input_data['ref_LM_input']['attention_mask'].chunk(num_chunk, 0)
        ):
            chunk_data = {
                'input_ids': chunk_input_ids,
                'attention_mask': chunk_attention_mask,
            }
            ref_logits_lst.append(self.forward(chunk_data))

        # 合并所有参考 logits
        ref_logits = torch.cat(ref_logits_lst, dim=0)  # 形状：(nc * batch_ref_cnt,)

        # 初始化与原始 prob_score 形状相同的参考概率矩阵
        ref_prob_score = torch.zeros_like(prob_score).to(prob_score.device)

        # ref_accum 用于定位当前样本在 ref_logits 中的偏移位置
        ref_accum = 0
        for tmp_i in range(bs):
            ref_cnt = input_data['ref_cnt'][tmp_i]  # 第 tmp_i 个样本的参考样本数量
            ref_accum += ref_cnt
            if ref_cnt == 0:
                continue

            # 提取该样本对应的参考 logits 并 reshape 成 (nc, ref_cnt)
            ref_logits_onesample = ref_logits[nc * (ref_accum - ref_cnt):nc * ref_accum].reshape(nc, ref_cnt)

            # 在每个选项维度上对参考样本 softmax，反映参考样本对各选项的支持度
            ref_prob_score_onesample = torch.softmax(ref_logits_onesample, dim=0)	# (nc, ref_cnt)

            # 取参考样本的平均概率，得到每个选项的参考评分
            ref_prob_score[tmp_i] = torch.mean(ref_prob_score_onesample, dim=1)	# (nc,)

        # 计算参考权重 ref_weight，只有有参考样本的样本才赋予非零权重
        ref_weight = torch.tensor(input_data['ref_cnt']).float().to(prob_score.device).reshape(-1, 1)
        ref_weight[ref_weight > 0] = 1 - self.args.CET_W0  # 有参考的样本分配一定比例的参考权重

        # 联合概率分数：融合原始问题预测概率与参考问题预测概率
        joint_prob_score = (1 - ref_weight) * prob_score + ref_weight * ref_prob_score

    # 计算最终的交叉熵损失（对 joint_prob_score 做 log，再计算负 log likelihood）
    loss = F.nll_loss(torch.log(joint_prob_score + 1e-10), labels)

    return loss, joint_prob_score  # 返回损失和联合预测分数

def evaluate_accuracy(dev_loader, model):
    n_corrects_acm_eval, n_samples_acm_eval = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        num_batch = len(dev_loader)
        for batch_idx in tqdm(list(range(num_batch)), total=num_batch, desc='Evaluation'):
            input_data = dev_loader[batch_idx]
            labels = input_data['example_label']

            logits = model.predict(input_data)

            bs = logits.shape[0]
            n_corrects = n_corrects = (logits.argmax(1) == labels).sum().item()
            n_corrects_acm_eval += n_corrects
            n_samples_acm_eval += bs

    ave_acc_eval = n_corrects_acm_eval / n_samples_acm_eval
    return ave_acc_eval


def set_random_seed(seed):
    if not seed is None:
        logger.info("Fix random seed")
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        logger.info("Use Random Seed")


def set_wandb(args):
    wandb_mode = "online" if args.use_wandb and (not args.debug) else "disabled"
    resume = (args.continue_train_from_check_path is not None) and (
                args.resume_id != "None" and args.resume_id is not None)
    args.wandb_id = args.resume_id if resume else wandb.util.generate_id()
    args.hf_version = transformers.__version__
    wandb_log = wandb.init(mode=wandb_mode, entity="your-entity", project="your-project", config=args,
                           name=args.run_name, resume="allow", id=args.wandb_id,
                           settings=wandb.Settings(start_method="thread"))
    logger.info('{0:>30}: {1}'.format("wandb id", args.wandb_id))
    return wandb_log


def main(args):
    set_random_seed(args.seed)
    print_system_info()
    print_basic_info(args)
    wandb_log = set_wandb(args)
    train(args, wandb_log)


def train(args, wandb_log):
    logger.info('=' * 71)
    logger.info('Start Training')
    logger.info('=' * 71)

    ###################################################################################################
    #   Get available GPU devices                                                                     #
    ###################################################################################################
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 1, 'No gpu avaliable!'

    # Note: Only using the pre-defined gpu_idx when debug; Otherwise, use CUDA_VISIBLE_DEVICES to specify the devices
    if (not args.use_wandb) and (args.gpu_idx is not None):
        gpu_idx = args.gpu_idx
        if isinstance(gpu_idx, int) or (isinstance(gpu_idx, str) and gpu_idx.isdigit()):  # gpu_idx.isdigit() 确保字符串内容为数字
            devices = torch.device(gpu_idx)
        else:
            raise Exception('Invalid gpu_idx {gpu_idx}')
    else:
        # logger.info('{0:>30}: {1}'.format('Visible GPU count',torch.cuda.device_count()))
        devices = torch.device(0)

    ###################################################################################################
    #   Build model                                                                                   #
    ###################################################################################################
    logger.info("Build model")
    if 'bert' in args.pretrain_model:
        model = BERT_basic(args)
    else:
        raise Exception('Invalid pretrain_model name %s' % args.pretrain_model)

    # Re-Init
    if args.is_ReInit:
        # First: Obtain a fully randomly initialized pretrained model
        random_init_pretrain_model = deepcopy(model.pretrain_model)
        # 对random_init_pretrain_model的所有层重新进行权重初始化
        random_init_pretrain_model.apply(
            random_init_pretrain_model._init_weights)  # using apply() to init each submodule recursively
        # Then: Set the top layers in the pretrained model
        if hasattr(random_init_pretrain_model.config, 'num_layers'):
            num_layers = random_init_pretrain_model.config.num_layers
        elif hasattr(random_init_pretrain_model.config, 'num_hidden_layers'):
            num_layers = random_init_pretrain_model.config.num_hidden_layers
        else:
            raise Exception('Cannot find number of layers in model.configs!!!')
        ignore_layers = [layer_i for layer_i in range(num_layers - args.ReInit_topk_layer)]
        reinit_lst = []  # 用于存储所有被重新初始化的参数名称

        for _name, _para in model.pretrain_model.named_parameters():  # 返回模型所有参数的名称和参数张量
            # Word embedding don't need initialization
            if 'shared' in _name or 'embeddings' in _name:
                continue
            # for bert
            if 'layer.' in _name:
                start_idx = _name.find('layer.') + len('layer.')  # 找到 'layer.' 的位置并加上长度
                end_idx = _name.find('.', start_idx)  # 从 start_idx 开始查找下一个 '.'
                layer_id = int(_name[start_idx:end_idx])
                if layer_id in ignore_layers:
                    continue

            model.pretrain_model.state_dict()[_name][:] = random_init_pretrain_model.state_dict()[_name][:]
            reinit_lst.append(_name)
        logger.info('Reinit modules: %s' % reinit_lst)
        del random_init_pretrain_model


    logger.info('Parameters statistics')
    params_statistic(model)

    ###################################################################################################
    #   Resume from checkpoint                                                                        #
    ###################################################################################################
    start_epoch = 0
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint.pt')
    if args.continue_train_from_check_path is not None and args.continue_train_from_check_path != 'None':
        logger.info("Resume from checkpoint %s" % args.continue_train_from_check_path)
        if torch.cuda.is_available():
            check = torch.load(args.continue_train_from_check_path)
        else:
            check = torch.load(args.continue_train_from_check_path, map_location=torch.device('cpu'))
        model_state_dict, _ = check
        model.load_state_dict(model_state_dict)
        model.train()

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################
    logger.info("Load dataset and dataloader")
    dataset = Basic_Dataloader(args, devices=devices)
    dev_loader = dataset.dev()
    test_loader = dataset.test()
    train_loader = dataset.train()

    ###################################################################################################
    #   Build Optimizer                                                                               #
    ###################################################################################################
    logger.info("Build optimizer")

    # You can use DataParallel here
    # model.pretrain_model = nn.DataParallel(model.pretrain_model, device_ids=(0,1))
    # model.pretrain_model.to(devices)

    optimizer, scheduler = get_optimizer(model, args, dataset)

    # ChildTune
    if args.optim == 'childtuningadamw' and args.ChildTuning_mode == 'ChildTuning-D':
        model = model.to(devices)
        gradient_mask = calculate_fisher(args, model, train_loader)
        optimizer.set_gradient_mask(gradient_mask)
        model = model.cpu()

    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################
    model.train()
    freeze_net(model.pretrain_model)
    logger.info("Freeze model.pretrain_model")

    model.to(devices)

    # record variables
    dev_acc = 0
    global_step, best_dev_epoch = 0, 0
    best_dev_acc, final_test_acc, best_test_acc = 0.0, 0.0, 0.0
    total_loss_acm, n_corrects_acm, n_samples_acm = 0.0, 0.0, 0.0
    best_dev_acc = dev_acc

    is_finish = False
    accumulate_batch_num = args.accumulate_batch_size // args.batch_size  # 128/8 = 16

    if args.is_CET:
        train_loader.generate_refs(model=model, load_cache=True)

    for epoch_id in trange(start_epoch, args.n_epochs, desc="Epoch"):

        model.epoch_idx = epoch_id

        if is_finish:
            break

        if epoch_id == args.unfreeze_epoch:
            unfreeze_net(model.pretrain_model)
            logger.info("Unfreeze model.pretrain_model")
        if epoch_id == args.refreeze_epoch:
            freeze_net(model.pretrain_model)
            logger.info("Freeze model.pretrain_model")

        model.train()

        start_time = time.time()

        num_batch = len(train_loader) - 1 if args.is_skip_last_batch else len(train_loader)  # 620

        for batch_id in tqdm(range(num_batch), total=num_batch, desc="Batch"):
            # load data for one batch
            input_data = train_loader.__getitem__(batch_id, is_skip_last_batch=args.is_skip_last_batch)
            labels = input_data['example_label']  # tensor([1, 2, 1, 1, 2, 2, 2, 1], device='cuda:0')
            bs = len(input_data['example_id'])  # 8

            if args.is_CET:
                loss, logits = model.compute_CET_loss(input_data, labels)
            else:
                loss, logits = model.compute_loss(input_data, labels)  # loss:1.3886  logits:tensor(8,4)

            total_loss_acm += loss.item() * bs
            loss.requires_grad_(True)
            loss.backward()

            n_corrects = (logits.detach().argmax(1) == labels).sum().item() if logits is not None else 0
            n_corrects_acm += n_corrects
            n_samples_acm += bs  # 8,16,...

            if (batch_id + 1) % accumulate_batch_num == 0 or batch_id == num_batch - 1:
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  # 更新学习率

            if (global_step + 1) % args.log_interval == 0:
                ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                logger.info('| step {:5} | lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step + 1,
                                                                                                   scheduler.get_last_lr()[
                                                                                                       0],
                                                                                                   total_loss_acm / n_samples_acm,
                                                                                                   ms_per_batch))

                if not args.debug:
                    wandb_log.log({"lr": scheduler.get_last_lr()[0], "train_loss": total_loss_acm / n_samples_acm,
                                   "train_acc": n_corrects_acm / n_samples_acm, "ms_per_batch": ms_per_batch},
                                  step=global_step + 1)

                total_loss_acm = 0.0
                n_samples_acm = n_corrects_acm = 0
                start_time = time.time()

            global_step += 1

        if epoch_id % args.eval_interval == 0:

            model.eval()
            dev_acc = evaluate_accuracy(dev_loader, model)

            test_acc = 0.0
            total_acc = []
            preds_path = os.path.join(args.save_dir, 'test_e{}_preds.csv'.format(epoch_id))
            with open(preds_path, 'w') as f_preds:
                with torch.no_grad():
                    num_batch = len(test_loader)
                    for batch_idx in tqdm(list(range(num_batch)), total=num_batch, desc='Testing'):
                        input_data = test_loader[batch_idx]
                        qids = input_data['example_id']
                        labels = input_data['example_label']
                        logits = model.predict(input_data)
                        predictions = logits.argmax(1)  # [bsize, ]
                        # preds_ranked = (-logits).argsort(1) #[bsize, n_choices]
                        for i, (qid, label, pred) in enumerate(zip(qids, labels, predictions)):
                            acc = int(pred.item() == label.item())
                            f_preds.write('{},{}\n'.format(qid, chr(ord('A') + pred.item())))
                            f_preds.flush()
                            total_acc.append(acc)
            test_acc = float(sum(total_acc)) / len(total_acc)

            best_test_acc = max(test_acc, best_test_acc)
            if epoch_id >= args.unfreeze_epoch:
                # update record variables
                if dev_acc >= best_dev_acc:
                    best_dev_acc = dev_acc
                    final_test_acc = test_acc
                    best_dev_epoch = epoch_id
                    if args.save_model:
                        model_path = os.path.join(args.save_dir, 'model.pt')
                        torch.save([model.state_dict(), args], model_path)
                        logger.info("model saved to %s" % model_path)
            else:
                best_dev_epoch = epoch_id

            logger.info('-' * 71)
            logger.info(
                '| epoch {:3} | step {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(epoch_id, global_step, dev_acc,
                                                                                         test_acc))
            logger.info('| best_dev_epoch {:3} | best_dev_acc {:7.4f} | final_test_acc {:7.4f} |'.format(best_dev_epoch,
                                                                                                         best_dev_acc,
                                                                                                         final_test_acc))
            logger.info('-' * 71)

            if not args.debug:
                wandb_log.log({"dev_acc": dev_acc, "dev_loss": dev_acc, "best_dev_acc": best_dev_acc,
                               "best_dev_epoch": best_dev_epoch}, step=global_step)
                if test_acc > 0:
                    wandb_log.log({"test_acc": test_acc, "test_loss": 0.0, "final_test_acc": final_test_acc},
                                  step=global_step)

            if args.save_check:
                training_dict = {'epoch': epoch_id, 'loss': loss,
                                 'model_state_dict': model.state_dict(),
                                 'optimizer_state_dict': optimizer.state_dict(),
                                 'scheduler_dict': scheduler.state_dict()}
                torch.save(training_dict, checkpoint_path)

            if epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                logger.info("After %d epoch no improving. Stop!" % (epoch_id - best_dev_epoch))
                logger.info("Best test accuracy: %s" % str(best_test_acc))
                logger.info("Final best test accuracy according to dev: %s" % str(final_test_acc))
                is_finish = True
                break
            model.train()

    ###################################################################################################
    #   Testing                                                                                       #
    ###################################################################################################
    if args.n_epochs <= 0:
        logger.info('n_epochs <= 0, start testing ...')
        model.eval()
        with torch.no_grad():
            dev_acc = evaluate_accuracy(dev_loader, model)
            test_acc = evaluate_accuracy(test_loader, model)
            logger.info('dev_acc {:7.4f} | test_acc {:7.4f}'.format(dev_acc, test_acc))


if __name__ == '__main__':
    args = get_args(is_save=True)
    logger = get_logger(args)
    main(args)
