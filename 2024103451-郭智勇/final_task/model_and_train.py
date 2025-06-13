import sys
sys.path.append("Encorder")
sys.path.append("Decorder")
sys.path.append("Datasets")
sys.path.append("数据集")
from torch import nn
import torch
from Encorder.Encoder import encoder
from Decorder.Decoder import decoder
from img_drop_embding import get_peach
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import datasets, transforms
device='cuda' if torch.cuda.is_available() else 'cpu'
# from value_process import val_list
from output import Generator
import numpy as np

# from all_dataset import train_labels,loss_labels,imgs,test_imgs,test_labels
from all_dataset import custom_dataset,test_imgs,test_labels
test_img=test_imgs
test_label=test_labels
from pos_embding import get_positional_encoding
from torch.optim.lr_scheduler import StepLR

import random

from swin import swin
swin_model=swin

#束搜索
def beam_search(encoder_decoder, test_imgs, start_token, end_token, beam_size=3, max_length=20):
    # 初始化候选序列
    sequences = [[[], 0.0, []]]  # 每个元素是一个元组 (序列, 总得分, 每个单词的得分列表)
    for _ in range(max_length):
        all_candidates = []
        # 扩展每个候选序列
        for seq, total_score, word_scores in sequences:
            if seq and seq[-1] == end_token:
                continue  # 如果序列已经结束，不再扩展
            # 使用模型预测下一个单词的概率
            if not seq:
                input_seq = [start_token]
            else:
                input_seq = seq
            input_seq = torch.tensor(input_seq, dtype=torch.int).to(device)
            output = encoder_decoder(test_imgs, input_seq,type='test')
            output = apply_repetition_penalty(output, input_seq, penalty=1.5)
            probs = torch.nn.functional.softmax(output, dim=-1)
            probs = probs[-1]  # 取最后一个单词的概率
            # 获取前 beam_size 个单词
            topk_probs, topk_indices = torch.topk(probs, beam_size)
            for p, idx in zip(topk_probs, topk_indices):
                new_seq = seq + [idx.item()]
                new_total_score = total_score + torch.log(p).item()
                new_word_scores = word_scores + [torch.log(p).item()]
                all_candidates.append((new_seq, new_total_score, new_word_scores))
        # 选择得分最高的 beam_size 个候选序列
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        # 检查是否所有序列都已结束
        if all(seq[-1] == end_token for seq, _, _ in sequences):
            break
    # 返回得分最高的序列及其每个单词的得分
    best_sequence, _, best_word_scores = sequences[0]

    # # 去除得分最低的单词
    if len(best_sequence) > 1:  # 如果序列长度大于1，才去除得分最低的单词
        # 找到得分最低的单词的索引
        min_score_index = best_word_scores.index(min(best_word_scores))
        # 去除得分最低的单词
        best_sequence = best_sequence[:min_score_index] + best_sequence[min_score_index + 1:]
        # 去除对应的得分
        best_word_scores = best_word_scores[:min_score_index] + best_word_scores[min_score_index + 1:]

    return best_sequence, best_word_scores

def apply_repetition_penalty(logits, generated_seq, penalty=1.2):
    for token in set(generated_seq):
        logits[:, token] /= penalty 
    return logits


seed=3407
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
np.random.seed(seed)
random.seed(seed)

#计算CER cer为 总dist/总长度
def edit_distance(str1, str2):
    if not str1:  # 如果str1为空字符串
        return len(str2)  # 编辑距离就是str2的长度
    if not str2:  # 如果str2为空字符串
        return len(str1)  # 编辑距离就是str1的长度

    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化边界条件
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # 填充dp表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1,      # 删除
                           dp[i][j - 1] + 1,      # 插入
                           dp[i - 1][j - 1] + cost)  # 替换

    return dp[m][n]


class EncoderDecoder(nn.Module):
    def __init__(self, encoder,decoder,source_emb,target_emb,generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = swin_model
        self.decoder = decoder
        self.generator = generator
        self.embed = source_emb
        self.target_emb = target_emb

    def forward(self, src, tgt, type='train'):
        x,mask=self.encode(src,type)
        target=self.decode(x,tgt,mask)
        cls=generator(target)
        return cls

    def encode(self, src,type='train'):
        res,mask=self.encoder(src,type=type)
        if mask is not None:
            mask=mask.unsqueeze(1).to(device)
        return res,mask

    def decode(self,src,tgt,src_mask):
        res=self.decoder(encoder_out=src,x=self.target_emb(tgt)+get_positional_encoding(self.target_emb(tgt).shape[-2],self.target_emb(tgt).shape[-1]).to(torch.float).to(device),src_mask=src_mask)
        return res


vocab_size=498
d_model=768
Embd=nn.Embedding(vocab_size,d_model)
generator=Generator(d_model=d_model,vocab_size=vocab_size).to(device)

encoder_decoder=EncoderDecoder(encoder,decoder,get_peach,Embd,generator).to(device)
encoder_decoder=encoder_decoder.to(device)

def char_index(list,x):
    for i in range(len(list)):
        if list[i]==x:
            return i
def creatStr(list):
    voc = []
    for item in list:
        voc.append(char_index(item, max(item)))
    return voc


#损失函数以及改进器
loss=nn.CrossEntropyLoss(ignore_index=129)
optimizer=torch.optim.AdamW(encoder_decoder.parameters(),lr=3e-5, betas=(0.9, 0.999),eps=1e-9,weight_decay=1e-5)#

# 使用StepLR调度器，每30个epoch衰减为原来的0.8倍
scheduler = StepLR(optimizer, step_size=30, gamma=0.8)

for p in encoder_decoder.parameters():#模型参数初始化
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

#训练集上的数据增强
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5,0.5], std=[0.5, 0.5, 0.5,0.5])
])


#测试集上的数据增强
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

pre=[]
res_pred=[]#测试集预测结果
best_cer=10000000
best_acc=0
print(encoder_decoder) #模型架构

# # 创建DataLoader实例
data_loader = DataLoader(custom_dataset, batch_size=16, shuffle=True)
for epoch in range(600):
    sun_loss = 0.0
    id=0
    for x,y,y1 in data_loader:   #imgs,loss_labels,train_labels
        x=x.to(device)
        y=y.to(device)
        y1=y1.to(device)
        id+=1
        imgs=[]
        for index,img in enumerate(x):
            image = transforms.ToPILImage()(x[index])
            imgs.append(image)
        res=encoder_decoder(imgs,y1).to(device)
        y = y.view(-1)  # 现在形状是 (8)
        l=loss(res,y.to(device)) #利用sum_loss_list计算损失
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        sun_loss +=l.item()
    print('epoch={}, avg_loss={:.2f}'.format(epoch, sun_loss))

    scheduler.step() 

    if epoch % 5 == 0:  # 每2个epoch测试一次模型
        print('*******' * 50)
        pre = []
        indexs = 0
        encoder_decoder.eval()
        sum_cer=0
        sum_acc=0
        sum_dist=0
        sum_char_num=0
        test_id=0
        true_num=0
        sum_num=0
        with torch.no_grad():
            for x, y in zip(test_img, test_label):
                test_id += 1
                test_imgs = [x]  

                y_pred,score = beam_search(encoder_decoder, test_imgs, start_token=490, end_token=491, beam_size=3,
                                     max_length=5)
                print('*' * 50)
                print(f"真实标签: {y[1:]}   长度: {len(y[1:])}")
                print(f"预测序列: {y_pred[:-1]}   长度: {len(y_pred[:-1])}")
                print(f"预测得分: {score[:-1]}   长度: {len(y_pred[:-1])}")
                count = sum(1 for x, y in zip(y_pred[:-1], y[1:]) if x == y)
                print(f"去除第二个单词之后匹配字符数: {count}")
                true_num += count
                sum_num += len(y[:-1])
                print(f"第 {test_id} 张图像")

                # 更新性能指标
                indexs += 1
                sum_acc = true_num/sum_num
                sum_dist += edit_distance(y_pred[:-1], y[:-1])
                sum_char_num += len(y[:-1])


            # 更新最佳性能指标
            if best_cer > (sum_dist / sum_char_num):
                best_cer = sum_dist / sum_char_num
            if sum_acc > best_acc:
                best_acc = sum_acc

            # 打印和保存结果
            print(f'最佳字符错误率 (CER): {best_cer}')
            print(f'最佳准确率: {best_acc}')
