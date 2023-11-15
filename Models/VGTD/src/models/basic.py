from __future__ import annotations

import logging
import time

import src
import torch
import torch.nn as nn
from hydra.utils import instantiate
from src.modules.embeddings import Embedding
from src.my_typing import *
from src.modules import transformer_attention

log = logging.getLogger('model')


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        attention_weights = self.softmax(torch.matmul(query, key.transpose(-2, -1)))
        weighted_values = torch.matmul(attention_weights, value)
        
        # 将加权平均后的值进行求和，得到1x768的特征向量
        feature = weighted_values.sum(dim=-2)
        return feature


class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainer: Trainer
        self.embedding: Embedding
        self.encoder: EncoderBase
        self.task: TaskBase
        self.datamodule: DataModule

        self._timing_decoding = False
        self._time = 0

    def setup(self, dm: DataModule):
        self.embedding = Embedding(src.g_cfg.embedding, dm)
        ## modify new
        # self.text_linear = self.Linear(1068,768)
        self.text_linear = nn.Linear(1068,768)
        num_heads = 8
        hidden_size =  768
        dropout = 0.1
        ff_size = hidden_size * 4
        num_layers = 3
        self.attention = transformer_attention.Encoder(
            lambda: transformer_attention.EncoderLayer(
            hidden_size,
            transformer_attention.MultiHeadedAttention(
                num_heads,
                hidden_size,
                dropout),
            transformer_attention.PositionwiseFeedForward(
                hidden_size,
                ff_size,
                dropout),                     
            dropout),
        hidden_size,
        num_layers,
        tie_layers=False)
        print(self.attention)
        self.fusion_way = "WeightedSum" # 1.ADD 2.SelfAttention 3.WeightedSum 4.?
        print("Fusion way:",self.fusion_way)
        self.beta = 0.5
        print(self.beta)
        if self.fusion_way == "SelfAttention":
            self.SelfAttention = SelfAttention(input_dim=hidden_size)
        self.encoder = instantiate(src.g_cfg.encoder, embedding=self.embedding)
        self.task = instantiate(src.g_cfg.task, encoder=self.encoder)
        self.datamodule = dm
        self.embedding.__dict__['bounded_model'] = self
        self.encoder.__dict__['bounded_model'] = self
        self.task.__dict__['bounded_model'] = self
        # log.info(self)

    def forward(self, x: InputDict, vp: VarPool, embed=None, encoded=None, return_all=False):
        if embed is None:
            if ("img_path_list" in x.keys()):
                word_embed, image_embed = self.embedding(x)
                new_word_embed = self.text_linear(word_embed)
                seq_len = x['seq_len']
                seq_len_list = seq_len.tolist()
                max_seq_len = max(seq_len_list)
                img_path_list = x['img_path_list']
                img_len_list = [len(i) for i in img_path_list]
                max_img_len = max(img_len_list)
                bs = new_word_embed.shape[0]
                attention_mask = torch.zeros(bs, max_seq_len, max_img_len).to(new_word_embed.device)
                # 填充掩码的有效部分为 1
                for i, (seq_len, image_num) in enumerate(zip(seq_len_list, img_len_list)):
                    attention_mask[i, :seq_len, :image_num] = 1
                # sequence_len = word_embed.shape[1]
                # image_embed = image_embed.unsqueeze(1).repeat(1, sequence_len, 1)
                #TTTTTTTTTT 下面3句都是新加的
                cross_embed = self.attention(new_word_embed, image_embed, image_embed, attention_mask)
                if self.fusion_way == "add":
                    embed = new_word_embed + cross_embed # 把文本embedding和corss attenttention embeedding相加，融合在一起！
                elif self.fusion_way == "SelfAttention":
                    new_word_embed = new_word_embed.unsqueeze(-2)
                    cross_embed = cross_embed.unsqueeze(-2)
                    temp_embed = torch.cat((new_word_embed, cross_embed),dim=-2)
                    embed = self.SelfAttention(temp_embed)
                elif self.fusion_way == "WeightedSum":
                    embed = self.beta * new_word_embed + (1.0 - self.beta) * cross_embed
                elif self.fusion_way == "concat":
                    embed = torch.cat((new_word_embed, cross_embed), dim=-1)
                else:
                    embed = cross_embed
                # embed = self.linear2(embed)
              #   fused_embedding = word_embed + cross_modal_attention
              #  last_hs = self.modal_interaction(fused_embedding, image_embed, x['seq_len'])
              #  last_hs = self.modal_interaction(word_embed, image_embed, x['seq_len'])  原来的句子
              #  embed = last_hs
            else:
                print("没有使用图片信息呀！！")
                embed = self.embedding(x)
        if encoded is None or embed is None:
            encoded = self.encoder(embed, vp)
        score = self.task.forward(encoded, vp)
        if return_all:
            return embed, encoded, score
        return score

    def loss(self, x: TensorDict, gold: InputDict, vp: VarPool) -> Tuple[Tensor, TensorDict]:
        return self.task.loss(x, gold, vp)

    def decode(self, x: TensorDict, vp: VarPool) -> AnyDict:
        if self._timing_decoding:
            torch.cuda.synchronize(device=None)
            start = time.time()
        result = self.task.decode(x, vp)
        if self._timing_decoding:
            torch.cuda.synchronize(device=None)
            self._time += time.time() - start
        return result

    def confidence(self, x: TensorDict, vp: VarPool, n: int = 1, gold: InputDict = None):
        return self.task.confidence(x, vp, n, gold)

    def normalize_embedding(self, now):
        self.embedding.normalize(now)

    def set_trainer(self, trainer: Trainer):
        self.trainer = trainer
        self.embedding.set_trainer(trainer)
        self.encoder.set_trainer(trainer)
        self.task.set_trainer(trainer)

    def preprocess_write(self, output: List[Dict[str, Any]]):
        return self.task.preprocess_write(output)

    def write_prediction(self, s: IOBase, predicts, dataset: DataSet, vocabs: Dict[str, Vocabulary]) -> IOBase:
        return self.task.write_prediction(s, predicts, dataset, vocabs)

    def set_varpool(self, vp: VarPool) -> VarPool:
        return self.task.set_varpool(vp)
    
    def Linear(self, in_features, out_features, bias=True):
        m = nn.Linear(in_features, out_features, bias)
        nn.init.xavier_uniform_(m.weight)
        if bias:
            nn.init.constant_(m.bias, 0.0)
        return m
