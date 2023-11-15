#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file  : metrics/ner.py
@author: xiaoya li
@contact : xiaoya_li@shannonai.com
@date  : 2021/01/14 16:13
@version: 1.0
@desc  :
"""

import torch
from typing import Any, List
from pytorch_lightning.metrics.metric import TensorMetric


class cws_SpanF1ForNER(TensorMetric):
    """
    compute span-level F1 scores for named entity recognition task.
    """
    def __init__(self, entity_labels: List[str] = None, reduce_group: Any = None, reduce_op: Any = None, save_prediction = False):
        super(cws_SpanF1ForNER, self).__init__(name="cws_span_f1_for_ner", reduce_group=reduce_group, reduce_op=reduce_op)
        self.num_labels = len(entity_labels)
        self.entity_labels = entity_labels
        self.tags2label = {label_idx : label_item for label_idx, label_item in enumerate(entity_labels)}
        self.save_prediction = save_prediction
        if save_prediction:
            self.pred_entity_lst = []
            self.gold_entity_lst = []


    def forward(self, pred_sequence_labels, gold_sequence_labels, sequence_mask=None):
        """
        Args:
            pred_sequence_labels: torch.LongTensor, shape of [batch_size, sequence_len]
            gold_sequence_labels: torch.LongTensor, shape of [batch_size, sequence_len]
            sequence_mask: Optional[torch.LongTensor], shape of [batch_size, sequence_len].
                        1 for non-[PAD] tokens; 0 for [PAD] tokens
        """

        cws_true_positive, cws_false_positive, cws_true_negative, cws_false_negative = 0, 0, 0, 0
        pred_sequence_labels = pred_sequence_labels.to("cpu").numpy().tolist()
        gold_sequence_labels = gold_sequence_labels.to("cpu").numpy().tolist()
        if sequence_mask is not None:
            sequence_mask = sequence_mask.to("cpu").numpy().tolist()
            # [1, 1, 1, 0, 0, 0]

        for item_idx, (pred_label_item, gold_label_item) in enumerate(zip(pred_sequence_labels, gold_sequence_labels)):
            if sequence_mask is not None:
                sequence_mask_item = sequence_mask[item_idx]
                try:
                    token_end_pos = sequence_mask_item.index(0) - 1 # before [PAD] always has an [SEP] token.
                except:
                    token_end_pos = len(sequence_mask_item) - 1
            else:
                token_end_pos = len(gold_label_item) - 1

            pred_label_item = [self.tags2label[tmp] for tmp in pred_label_item[1:token_end_pos]]
            gold_label_item = [self.tags2label[tmp] for tmp in gold_label_item[1:token_end_pos]]

            pred_entities = transform_entity_bmes_labels_to_spans(pred_label_item)
            gold_entities = transform_entity_bmes_labels_to_spans(gold_label_item)

            if self.save_prediction:
                # self.pred_entity_lst.append(pred_entities)
                # self.gold_entity_lst.append(gold_entities)
                self.pred_entity_lst.append(pred_label_item)
                self.gold_entity_lst.append(gold_label_item)

          
            cws_tp, cws_fp, cws_fn = count_cws_confusion_matrix(pred_entities, gold_entities)

            cws_true_positive += cws_tp
            cws_false_positive += cws_fp
            cws_false_negative += cws_fn

            

      
        cws_batch_confusion_matrix = torch.LongTensor([cws_true_positive, cws_false_positive, cws_false_negative])
        return cws_batch_confusion_matrix

    def compute_f1_using_confusion_matrix(self, true_positive, false_positive, false_negative, prefix="dev"):
        """
        compute f1 scores.
        Description:
            f1: 2 * precision * recall / (precision + recall)
                - precision = true_positive / true_positive + false_positive
                - recall = true_positive / true_positive + false_negative
        Returns:
            precision, recall, f1
        """
        precision = true_positive / (true_positive + false_positive + 1e-13)
        recall = true_positive / (true_positive + false_negative + 1e-13)
        f1 = 2 * precision * recall / (precision + recall + 1e-13)

        if self.save_prediction and prefix == "dev":
            self.pred_entity_lst = []
            self.gold_entity_lst = []

        if self.save_prediction and prefix == "test":
            entity_tuple = (self.gold_entity_lst, self.pred_entity_lst)
            return precision, recall, f1, entity_tuple

        return precision, recall, f1


def count_cws_confusion_matrix(pred_entities, gold_entities):
    true_positive, false_positive, false_negative = 0, 0, 0
    pred_entities = [('SEG', s[1]) if isinstance(s, tuple) else 'SEG' for s in pred_entities]
    gold_entities = [('SEG', s[1]) if isinstance(s, tuple) else 'SEG' for s in gold_entities]
    for span_item in pred_entities:
        if span_item in gold_entities:
            true_positive += 1
            gold_entities.remove(span_item)
        else:
            false_positive += 1
    # these entities are not predicted.
    for span_item in gold_entities:
        false_negative += 1

    return true_positive, false_positive, false_negative

def transform_entity_bmes_labels_to_spans(label_sequence, classes_to_ignore=None):
    """
    Given a sequence of BMES-{entity type} labels, extracts spans.
    """
    spans = []
    classes_to_ignore = classes_to_ignore or []
    index = 0
    while index < len(label_sequence):
        label = label_sequence[index]
        if label[0] == "S":
            spans.append((label.split("-")[1], (index, index)))
        elif label[0] == "B":
            sign = 1
            start = index
            start_cate = label.split("-")[1]
            while label[0] != "E":
                index += 1
                if index >= len(label_sequence):
                    spans.append((start_cate, (start, start)))
                    sign = 0
                    break
                label = label_sequence[index]
                if not (label[0] == "M" or label[0] == "E"):
                    spans.append((start_cate, (start, start)))
                    sign = 0
                    break
                if label.split("-")[1] != start_cate:
                    spans.append((start_cate, (start, start)))
                    sign = 0
                    break
            if sign == 1:
                spans.append((start_cate, (start, index)))
        else:
            if label != "O":
                pass
        index += 1
    return [span for span in spans if span[0] not in classes_to_ignore]


def cws_evaluate_OOV(y_pred_list, y_list, sentence_list, word2id, tag2index):
    cor_num = 0
    yt_wordnum = 0
    index2tag = {index: tag for tag, index in tag2index.items()}
    for y_pred, y, sentence in zip(y_pred_list, y_list, sentence_list):
        start = 0
        for i in range(len(y)):
            tag = index2tag[y[i].item()]
            if "E" in tag or "S" in tag:
            # if y[i] == tag2index["E"] or y[i] == tag2index["S"]:
                #import pdb;pdb.set_trace()
                word = ''.join(sentence[0][start:i+1])
                if word in word2id:
                    start = i + 1
                    continue
                flag = True
                yt_wordnum += 1
                for j in range(start, i+1):
                    if y[j] != y_pred[j]:
                        flag = False
                if flag:
                    cor_num += 1
                start = i + 1

    OOV = cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    return OOV