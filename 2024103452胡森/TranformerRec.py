import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class FeedForward(nn.Module):
    """
        hidden_size: 隐藏层大小
        inner_size: 中间层大小
        hidden_dropout_prob: Dropout概率
        layer_norm_eps: LayerNorm的epsilon值
    """
    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, layer_norm_eps
    ):
        super(FeedForward, self).__init__()
        # 第一层线性变换
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        # GELU激活函数
        self.Gelu = nn.GELU()
        # 第二层线性变换
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        # LayerNorm层
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        # Dropout层
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.dense_1.weight)
        nn.init.xavier_uniform_(self.dense_2.weight)
        nn.init.normal_(self.dense_1.bias, std=1e-6)
        nn.init.normal_(self.dense_2.bias, std=1e-6)

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.Gelu(hidden_states)
        
        hidden_states = self.dense_2(hidden_states)
        # Dropout
        hidden_states = self.dropout(hidden_states)
        
        # 残差
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
    

class MultiHeadAttention(nn.Module):
    """
        n_heads: 注意力头数
        hidden_size: 隐藏层大小
        hidden_dropout_prob: 隐藏层Dropout概率
        attn_dropout_prob: 注意力Dropout概率
        layer_norm_eps: LayerNorm的epsilon值
    """
    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({n_heads})"
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        # 查询、键、值的线性变换层
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # Softmax和Dropout层
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        # 输出层
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        
        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.normal_(self.query.bias, std=1e-6)
        nn.init.normal_(self.key.bias, std=1e-6)
        nn.init.normal_(self.value.bias, std=1e-6)
        nn.init.normal_(self.dense.bias, std=1e-6)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)
        
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask
        
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    """
    参数:
        n_heads: 注意力头数
        hidden_size: 隐藏层大小
        intermediate_size: 前馈网络中间层大小
        hidden_dropout_prob: 隐藏层Dropout概率
        attn_dropout_prob: 注意力Dropout概率
        layer_norm_eps: LayerNorm的epsilon值
    """
    def __init__(self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size, intermediate_size, hidden_dropout_prob, layer_norm_eps
        )
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(
            self.attention_layer_norm(hidden_states), attention_mask
        )
        attention_output = attention_output + hidden_states
        
        feedforward_output = self.feed_forward(
            self.ffn_layer_norm(attention_output)
        )
        feedforward_output = feedforward_output + attention_output
        
        return feedforward_output


class TransformerEncoder(nn.Module):
    """
        n_layers: Transformer层数 (默认: 2)
        n_heads: 注意力头数 (默认: 2)
        hidden_size: 隐藏层大小 (默认: 64)
        inner_size: 前馈网络中间层大小 (默认: 256)
        hidden_dropout_prob: 隐藏层Dropout概率 (默认: 0.5)
        attn_dropout_prob: 注意力Dropout概率 (默认: 0.5)
        layer_norm_eps: LayerNorm的epsilon值 (默认: 1e-12)
    """
    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        layer_norm_eps=1e-12,
    ):
        super(TransformerEncoder, self).__init__()
        
        layer = TransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.inner_size = inner_size

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        
        hidden_states = self.final_layer_norm(hidden_states)
        
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            
        return all_encoder_layers
    
    def get_config(self):
        return {
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "hidden_size": self.hidden_size,
            "inner_size": self.inner_size
        }


class TransformerRec(nn.Module):
    def __init__(self, config, dataset):
        super(TransformerRec, self).__init__()
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        
        self.n_items = dataset.num_items
        self.max_seq_length = dataset.max_seq_length
        
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        self.transformer_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            layer_norm_eps=self.layer_norm_eps,
        )
        
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        self.apply(self._init_weights)
        
        self._log_model_info()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _log_model_info(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized TransformerRec model with {total_params:,} trainable parameters")
        print(f"Model configuration: n_layers={self.n_layers}, n_heads={self.n_heads}, "
              f"hidden_size={self.hidden_size}, inner_size={self.inner_size}")
        print(f"Dropout rates: hidden={self.hidden_dropout_prob}, attn={self.attn_dropout_prob}")

    def get_attention_mask(self, item_seq):
        # 创建基础掩码（0表示有效位置，1表示填充位置）
        attention_mask = (item_seq == 0).float()
        # 扩展维度用于多头注意力
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # 转换为较大的负数，使softmax后接近0
        extended_attention_mask = extended_attention_mask * -10000.0
        return extended_attention_mask

    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(1, gather_index)
        return output_tensor.squeeze(1)

    def forward(self, item_seq, item_seq_len):
        # 生成位置ID
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        
        # 位置嵌入
        position_embedding = self.position_embedding(position_ids)
        # 项目嵌入
        item_emb = self.item_embedding(item_seq)
        # 组合嵌入
        input_emb = item_emb + position_embedding
        # 层归一化和Dropout
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # 生成注意力掩码
        extended_attention_mask = self.get_attention_mask(item_seq)

        # 通过Transformer编码器
        trm_output = self.transformer_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        
        # 获取序列最后位置的输出
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        """计算损失"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # 获取序列表示
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            # 计算正负样本分数
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            # 计算BPR损失
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            # 获取所有项目嵌入
            test_item_emb = self.item_embedding.weight
            # 计算所有项目的分数
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            # 计算交叉熵损失
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        # 获取序列表示
        seq_output = self.forward(item_seq, item_seq_len)
        # 获取测试项目嵌入
        test_item_emb = self.item_embedding(test_item)
        # 计算分数
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # 获取序列表示
        seq_output = self.forward(item_seq, item_seq_len)
        # 获取所有项目嵌入
        test_items_emb = self.item_embedding.weight
        # 计算所有项目的分数
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'n_layers': self.n_layers,
                'n_heads': self.n_heads,
                'hidden_size': self.hidden_size,
                'inner_size': self.inner_size,
                'hidden_dropout_prob': self.hidden_dropout_prob,
                'attn_dropout_prob': self.attn_dropout_prob,
                'layer_norm_eps': self.layer_norm_eps,
                'initializer_range': self.initializer_range,
                'loss_type': self.loss_type,
                'n_items': self.n_items,
                'max_seq_length': self.max_seq_length
            }
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path, dataset):
        checkpoint = torch.load(path)
        config = checkpoint['config']
        config["n_items"] = checkpoint['config']['n_items']
        config["max_seq_length"] = checkpoint['config']['max_seq_length']
        
        model = cls(config, dataset)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
        return model

    def get_attention_weights(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        
        extended_attention_mask = self.get_attention_mask(item_seq)
        
        all_attention_weights = []
        hidden_states = input_emb
        
        for layer in self.transformer_encoder.layer:
            attention_output = layer.multi_head_attention(
                layer.attention_layer_norm(hidden_states), extended_attention_mask
            )
            all_attention_weights.append(None)
            
            feedforward_output = layer.feed_forward(
                layer.ffn_layer_norm(attention_output)
            )
            hidden_states = feedforward_output + attention_output
            
        return all_attention_weights


class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma
        
    def forward(self, pos_scores, neg_scores):
        diff = pos_scores - neg_scores
        # 计算损失
        loss = -torch.log(torch.sigmoid(diff) + self.gamma)
        return loss.mean()





def model_summary(model):
    print("Model Summary:")
    print("-" * 80)
    print(f"{'Layer (type)':<30} {'Output Shape':<20} {'Param #':<15}")
    print("=" * 80)
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        total_params += num_params
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        trainable_params += trainable
        
        output_shape = "varies"
        if hasattr(module, 'weight'):
            if isinstance(module.weight, torch.Tensor):
                output_shape = str(list(module.weight.shape))
        
        print(f"{name:<30} {output_shape:<20} {num_params:<15,}")
    
    print("=" * 80)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("-" * 80)