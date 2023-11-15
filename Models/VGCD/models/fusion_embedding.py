#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : glyce_embedding.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/8/23 10:40
@version: 1.0
@desc  : 【char embedding】+【pinyin embedding】+【glyph embedding】 = fusion embedding
"""
import os

import torch
from torch import nn

from models.glyph_embedding import GlyphEmbedding
from models.pinyin_embedding import PinyinEmbedding
from models.extract_img_feats import *

class FusionBertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position, glyph, pinyin and token_type embeddings.
    """

    def __init__(self, config):
        super(FusionBertEmbeddings, self).__init__()
        config_path = os.path.join(config.name_or_path, 'config')
        font_files = []
        for file in os.listdir(config_path):
            if file.endswith(".npy"):
                font_files.append(os.path.join(config_path, file))
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.pinyin_embeddings = PinyinEmbedding(embedding_size=128, pinyin_out_dim=config.hidden_size,
                                                 config_path=config_path)
        self.glyph_embeddings = GlyphEmbedding(font_npy_files=font_files)

        # self.LayerNorm is not snake-cased to stick with TensorFlow models variable name and be able to load
        # any TensorFlow checkpoint file
        self.glyph_map = nn.Linear(1728, config.hidden_size)
        self.map_fc = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # image embedding
        self.img_embedding = Img_Embedding()



    def forward(self, input_ids=None, pinyin_ids=None, batch_images_list=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # get char embedding, pinyin embedding and glyph embedding
        word_embeddings = inputs_embeds  # [bs,l,hidden_size]
        pinyin_embeddings = self.pinyin_embeddings(pinyin_ids)  # [bs,l,hidden_size]
        glyph_embeddings = self.glyph_map(self.glyph_embeddings(input_ids))  # [bs,l,hidden_size]
        # fusion layer 将三个embedding结合在一起，然后通过fusion layer。
        concat_embeddings = torch.cat((word_embeddings, pinyin_embeddings, glyph_embeddings), 2)
        inputs_embeds = self.map_fc(concat_embeddings)
        # 将通过fusion layer后的embedding，与position embedding，token type embedding结合在一起。
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        if batch_images_list:
            text_embeddings = embeddings
            image_embeddings, img_len_list = self.img_embedding(batch_images_list)
            return text_embeddings, image_embeddings, img_len_list
        else:
            text_embeddings = embeddings
            return embeddings
