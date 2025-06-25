## 基于因果推理从预训练语言模型中保留常识知识用于常识问答

### 1 模型中文描述

该方法的目的是基于因果推理从预训练语言模型中保留常识知识并用于常识问答任务，叫做因果效应微调。该方法通过最大化因果效应来解决预训练语言模型的灾难性遗忘问题。我们首先将常识问答任务定义如下：给定一个包含 $N$ 个样本的数据集 $\left\{ \left(q^{(i)}, a^{(i)}, \{o_j^{(i)}\}_j \right) \right\}_{i}^{N}$，我们训练最佳模型，对于给定的问题 $q^{(i)}$，从选项 $\{o_j^{(i)}\}$ 中选择正确答案 $a^{(i)}$。对于给定的问题$q^{(i)}$，我们基于因果推理中的碰撞效应理论，检索与问题$q^{(i)}$最相似的K个问题，由于共享相同答案的问题在PLMs中共享相同的常识知识，在检索KNN时，我们使用真实答案之间的相似度来度量距离，而不是使用问题本身，通过答案之间的相似度来评估问题相似度。我们将第 $i$ 个样本的 KNN 的输入定义为$x^{(i,k)} = q^{(i,k)} \mathbin\Vert o_1^{(i)} \mathbin\Vert \cdots \mathbin\Vert o_j^{(i)}$。我们将原问题$q^{(i)}$与其KNN一同输入到PLM中，模型需要在原问题$q^{(i)}$的选项中预测正确的答案。

### 2 模型图

![image-20250521182218086](D:\OneDrive\项目实践\images\image-20250521182218086.png)

上图展示了因果效应微调的方法。其中，$x^{(i)}$ 是锚样本，$h_0^{(i)}$ 是由预训练模型提取的隐藏特征，$\{x^{(i,1)}, x^{(i,2)}, x^{(i,3)}\}$ 是 $x^{(i)}$ 的 KNN。我们通过在 $x^{(i)}$ 上施加碰撞效应以保留旧有知识。在微调过程中，“红色”知识由于存在碰撞效应得以被保留，而“蓝色”知识由于缺乏碰撞效应从而被遗忘。

### 3 实验

#### 3.1 数据集

- **CommonsenseQA (CSQA)：**此数据集包含需要常识推理的问题。示例如下：

  ```json
  {"answerKey": "C", "id": "23505889b94e880c3e89cff4ba119860", "question": {"question_concept": "fox", "choices": [{"label": "A", "text": "pretty flowers."}, {"label": "B", "text": "hen house"}, {"label": "C", "text": "natural habitat"}, {"label": "D", "text": "storybook"}, {"label": "E", "text": "dense forest"}], "stem": "The fox walked from the city into the forest, what was it looking for?"}, "statements": [{"label": false, "statement": "The fox walked from the city into the forest, pretty flowers was it looking for."}, {"label": false, "statement": "The fox walked from the city into the forest, hen house was it looking for."}, {"label": true, "statement": "The fox walked from the city into the forest, natural habitat was it looking for."}, {"label": false, "statement": "The fox walked from the city into the forest, storybook was it looking for."}, {"label": false, "statement": "The fox walked from the city into the forest, dense forest was it looking for."}]}
  ```

- **OpenBookQA. (OBQA)：**此数据集需要用基本的科学知识进行推理。示例如下：

  ```json
  {"id": "7-637", "question": {"stem": "When the weather changes as it does from Christmas to Easter,", "choices": [{"text": "the air may chill", "label": "A"}, {"text": "the ground may freeze", "label": "B"}, {"text": "the plants may die", "label": "C"}, {"text": "the ground may warm", "label": "D"}]}, "answerKey": "D"}
  ```

#### 3.2 评价标准

$\text{准确率} = \frac{\text{预测正确的样本数量}}{\text{总样本数量}}$

#### 3.3 参数设定

```yaml
n_epochs: 50	# 训练的总轮数
max_epochs_before_stop: 10	# 早停策略
accumulate_batch_size: 128	# 累积梯度的目标批次大小
batch_size: 8	# 每次前向传播的实际批次大小
eval_batch_size: 8	# 评估时使用的批次大小
lr: 1e-5	# 学习率
warmup_steps: 150	# 学习率预热步数
optim: radam	# 优化器类型
weight_decay: 0.01  # 权重衰减
pretrain_model: roberta-large	# 使用的预训练语言模型
CET_topk: 5	# KNN数量
CET_W0: 0.9	# 目标问题得分所占权重
```

#### 3.4 模型参数量

```
355.361 M
```

#### 3.5 实验结果

3.5.1 主要结果

| 数据集          | CSQA  | OBQA  |
| --------------- | ----- | ----- |
| **准确率（%）** | 76.82 | 70.76 |

3.5.2 训练曲线（以 CSQA 为例）

![image-20250526132430372](D:\OneDrive\项目实践\images\image-20250526132430372.png)

### 4 核心代码

4.1 训练模型

```python
# 如果启用因果效应微调，生成参考样本（可加载缓存）
if args.is_CET:
    train_loader.generate_refs(model=model, load_cache=True)

# 遍历所有 epoch
for epoch_id in trange(start_epoch, args.n_epochs, desc="Epoch"):

    model.epoch_idx = epoch_id  # 记录当前 epoch 编号

    if is_finish:
        break  # 提前结束训练（如 early stopping）

    model.train()  # 设置模型为训练模式
    start_time = time.time()  # 记录当前 batch 开始时间

    # 决定是否跳过最后一个 batch（如果不满 accumulate_batch_size）
    num_batch = len(train_loader) - 1 if args.is_skip_last_batch else len(train_loader)

    # 遍历每个 batch
    for batch_id in tqdm(range(num_batch), total=num_batch, desc="Batch"):
        # 加载一个 batch 的数据（通过自定义的 __getitem__ 方法）
        input_data = train_loader.__getitem__(batch_id, is_skip_last_batch=args.is_skip_last_batch)
        labels = input_data['example_label']  # 获取标签，例如 tensor([1, 2, 1, ...])
        bs = len(input_data['example_id'])  # 当前 batch 大小，如 8

        # 根据不同训练策略（CET、BSS、R3F）调用相应的 loss 计算函数
        if args.is_CET:
            loss, logits = model.compute_CET_loss(input_data, labels)
        else:
            loss, logits = model.compute_loss(input_data, labels)

        # 累加损失（乘以样本数用于计算平均）
        total_loss_acm += loss.item() * bs

        # 使 loss 参与梯度计算并反向传播
        loss.requires_grad_(True)
        loss.backward()

        # 统计当前 batch 的预测正确数量
        n_corrects = (logits.detach().argmax(1) == labels).sum().item() if logits is not None else 0
        n_corrects_acm += n_corrects
        n_samples_acm += bs  # 累计样本数

        # 达到梯度累积数量，或是最后一个 batch，则进行参数更新
        if (batch_id + 1) % accumulate_batch_num == 0 or batch_id == num_batch - 1:
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 梯度裁剪
            optimizer.step()        # 更新模型参数
            optimizer.zero_grad()   # 清空梯度缓存
            scheduler.step()        # 更新学习率

        # 每隔 log_interval 步记录一次日志
        if (global_step + 1) % args.log_interval == 0:
            ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval  # 每 batch 所花时间（毫秒）
            logger.info('| step {:5} | lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |'.format(
                global_step + 1,
                scheduler.get_last_lr()[0],
                total_loss_acm / n_samples_acm,
                ms_per_batch
            ))

            # 清空累计指标以便下一次日志更新
            total_loss_acm = 0.0
            n_samples_acm = n_corrects_acm = 0
            start_time = time.time()

        global_step += 1  # 训练步数 +1

```

4.2 寻找参考问题

```python
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

```

4.3 计算联合损失

```python
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

```
