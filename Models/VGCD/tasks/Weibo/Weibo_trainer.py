#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file  : Weibo_trainer.py
@author: xiaoya li
@contact : xiaoya_li@shannonai.com
@date  : 2021/06/30 17:28
@version: 1.0
@desc  :
"""

import os
import re
import json
import argparse
import logging
from functools import partial
from collections import namedtuple

from datasets.collate_functions import collate_to_max_length
from datasets.weibo_ner_dataset import WeiboNERDataset
from models.modeling_glycebert import GlyceBertForTokenClassification
from utils.random_seed import set_random_seed
from metrics.ner import SpanF1ForNER
from metrics.ner_pos import cws_SpanF1ForNER

# enable reproducibility
# https://pytorch-lightning.readthedocs.io/en/latest/trainer.html
set_random_seed(2333)

import torch
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class WeiboTask(pl.LightningModule):
#该模块继承自pl.LightningModule，它实现了PyTorch Lightning中规定的训练和验证等步骤。

#这个模块实现了NER任务的训练、验证和测试过程，并且能够将结果保存到文件中以供进一步分析。
# 它结合了PyTorch Lightning的训练框架和Hugging Face Transformers库的BERT模型，方便实现微博NER任务的快速开发和调试。

    def __init__(self, args: argparse.Namespace):
        """Initialize a model, tokenizer and config"""
        #初始化函数，用于创建并配置模型。参数args是一个命名空间（argparse.Namespace）对象，用于保存训练时的参数配置。

        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        else:
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)

        self.entity_labels = WeiboNERDataset.get_labels(os.path.join(args.data_dir, "ner_labels.txt"))
        self.bert_dir = args.bert_path
        self.num_labels = len(self.entity_labels)
        self.bert_config = BertConfig.from_pretrained(self.bert_dir, output_hidden_states=True,
                                                      num_labels=self.num_labels,
                                                      hidden_dropout_prob=self.args.hidden_dropout_prob)
        #根据预训练模型的路径、是否输出隐藏状态、标签数量和隐藏层丢弃概率等参数，创建并配置一个BERT模型的配置对象。
        
        self.model = GlyceBertForTokenClassification.from_pretrained(self.bert_dir,
                                                                     config=self.bert_config,
                                                                     mlp=False if self.args.classifier=="single" else True)
        #根据预训练模型的路径、配置对象和分类器类型，创建并配置一个预训练的GlyceBERT模型。

        self.ner_evaluation_metric = SpanF1ForNER(entity_labels=self.entity_labels, save_prediction=self.args.save_ner_prediction)
        #它用于计算命名实体识别任务中的跨度级别（span-level）的F1分数。

        self.cws_ner_evaluation_metric = cws_SpanF1ForNER(entity_labels=self.entity_labels, save_prediction=self.args.save_ner_prediction)
        
        format = '%(asctime)s - %(name)s - %(message)s'
        #定义了日志记录的格式字符串。它包含三个占位符，分别表示记录的时间、记录器的名称和记录的消息。

        logging.basicConfig(format=format, filename=os.path.join(self.args.save_path, "eval_result_log.txt"), level=logging.INFO)
        #配置日志记录器的基本设置。

        self.result_logger = logging.getLogger(__name__)
        #获取一个名为__name__的日志记录器。

        self.result_logger.setLevel(logging.INFO)
        #设置日志记录器的级别为INFO，表示只记录INFO级别及以上的日志消息。

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        #配置优化器和学习率调度器，并设置线性学习率预热和衰减。

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.args.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), lr=self.args.lr, eps=self.args.adam_epsilon, )
        elif self.args.optimizer == "torch.adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)
        else:
            raise ValueError("Please import the Optimizer first. ")
        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (
                self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proportion * t_total)
        if self.args.no_lr_scheduler:
            return [optimizer]
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, pinyin_ids, batch_images_list):
        #模型前向传播函数。

        attention_mask = (input_ids != 0).long()
        return self.model(input_ids, pinyin_ids, batch_images_list, attention_mask=attention_mask)

    def compute_loss(self, logits, labels, loss_mask=None):
        #计算交叉熵损失。
        """
        Desc:
            compute cross entropy loss
        Args:
            logits: FloatTensor, shape of [batch_size, sequence_len, num_labels]
            labels: LongTensor, shape of [batch_size, sequence_len, num_labels]
            loss_mask: Optional[LongTensor], shape of [batch_size, sequence_len].
                1 for non-PAD tokens, 0 for PAD tokens.
        """
        loss_fct = CrossEntropyLoss()
        if loss_mask is not None:
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        # 训练过程中的单步函数，用于计算训练损失。接收一个批次数据batch和批次索引batch_idx，返回损失值和其他需要记录的日志信息。
        batch_images_list = batch[1]
        input_ids, pinyin_ids, labels = batch[0]
        #从批次数据batch中解包得到输入ID、拼音ID和标签。

        loss_mask = (input_ids != 0).long()
        #创建一个损失掩码loss_mask，将非零元素设为1，零元素设为0。

        batch_size, seq_len = input_ids.shape
        #获取输入ID的形状，得到批次大小batch_size和序列长度seq_len。

        pinyin_ids = pinyin_ids.view(batch_size, seq_len, 8)
        #调整拼音ID的形状，将其视为形状为(batch_size, seq_len, 8)的三维张量。
        #这里假设拼音ID是一个形状为(batch_size, seq_len*8)的二维张量，将其转换为三维张量，每个序列位置有8个拼音特征。

        sequence_logits = self.forward(input_ids=input_ids, pinyin_ids=pinyin_ids, batch_images_list=batch_images_list)[0]
        #调用模型的前向传播方法forward，传入输入ID和拼音ID，获取序列的预测logits。

        loss = self.compute_loss(sequence_logits, labels, loss_mask=loss_mask)
        #计算损失值，调用compute_loss方法，传入序列logits、标签和损失掩码，获取损失值。

        tf_board_logs = {
            "train_loss": loss,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"]
        }
        #创建一个字典tf_board_logs，包含需要记录到TensorBoard的日志信息。这里记录了训练损失和学习率。

        return {"loss": loss, "log": tf_board_logs}
        #返回一个字典，包含损失值和需要记录的日志信息。这个字典将被传递给训练器（trainer），用于日志记录和优化器更新。

    def validation_step(self, batch, batch_idx):
        #验证过程中的单步函数，用于计算验证损失和评估指标。接收一个批次数据batch和批次索引batch_idx，返回验证损失和评估指标。
        batch_images_list = batch[1]
        input_ids, pinyin_ids, gold_labels = batch[0]
        batch_size, seq_len = input_ids.shape
        loss_mask = (input_ids != 0).long()
        pinyin_ids = pinyin_ids.view(batch_size, seq_len, 8)
        sequence_logits = self.forward(input_ids=input_ids, pinyin_ids=pinyin_ids, batch_images_list=batch_images_list)[0]
        loss = self.compute_loss(sequence_logits, gold_labels, loss_mask=loss_mask)
        probabilities, argmax_labels = self.postprocess_logits_to_labels(sequence_logits.view(batch_size, seq_len, -1))
        confusion_matrix = self.ner_evaluation_metric(argmax_labels, gold_labels, sequence_mask=loss_mask)
        return {"val_loss": loss, "confusion_matrix": confusion_matrix}

    def validation_epoch_end(self, outputs):
        #验证过程结束后的回调函数，用于计算平均验证损失和评估指标。接收一个包含所有validation_step返回值的列表outputs，返回一个包含验证损失和评估指标的字典。
        # avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        # confusion_matrix = torch.stack([x[f"confusion_matrix"] for x in outputs]).sum(0)
        # all_true_positive, all_false_positive, all_false_negative = confusion_matrix
        # precision, recall, f1 = self.ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative)

        # self.result_logger.info(f"EVAL INFO -> current_epoch is: {self.trainer.current_epoch}, current_global_step is: {self.trainer.global_step} ")
        # self.result_logger.info(f"EVAL INFO -> valid_f1 is: {f1}")
        # tensorboard_logs = {"val_loss": avg_loss, "val_f1": f1, }
        # return {"val_loss": avg_loss, "val_log": tensorboard_logs, "val_f1": f1, "val_precision": precision, "val_recall": recall}

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        confusion_matrix = torch.stack([x["confusion_matrix"] for x in outputs]).sum(0)
        all_true_positive, all_false_positive, all_false_negative = confusion_matrix
        
        if self.args.save_ner_prediction:
            precision, recall, f1, entity_tuple = self.ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative, prefix="dev")    
            gold_entity_lst, pred_entity_lst = entity_tuple
            self.save_predictions_to_file(gold_entity_lst, pred_entity_lst, prefix="dev")
        else:
            precision, recall, f1 = self.ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative)

        self.result_logger.info(f"EVAL INFO -> current_epoch is: {self.trainer.current_epoch}, current_global_step is: {self.trainer.global_step} ")
        self.result_logger.info(f"EVAL INFO -> valid_f1 is: {f1}, precision is: {precision}, recall is: {recall}")
        print("=="*50)
        print("正在评价dev上的结果：")
        print((f"EVAL INFO -> current_epoch is: {self.trainer.current_epoch}, current_global_step is: {self.trainer.global_step} "))
        print(f"DEV RESULT -> DEV F1: {f1}, Precision: {precision}, Recall: {recall} ")
        print("=="*50)
        tensorboard_logs = {"val_loss": avg_loss, "val_f1": f1, }
        return {"val_loss": avg_loss, "val_log": tensorboard_logs, "val_f1": f1, "val_precision": precision, "val_recall": recall}

    def train_dataloader(self,) -> DataLoader:
        #获取训练数据加载器。
        return self.get_dataloader("train")

    def val_dataloader(self, ) -> DataLoader:
        #: 获取验证数据加载器。
        return self.get_dataloader("dev")

    def _load_dataset(self, prefix="test"):
        dataset = WeiboNERDataset(directory=self.args.data_dir, prefix=prefix,
                                    vocab_file=os.path.join(self.args.bert_path, "vocab.txt"),
                                    max_length=self.args.max_length,
                                    config_path=os.path.join(self.args.bert_path, "config"),
                                    file_name=self.args.file_name, task_name=self.args.task_name)

        return dataset

    def get_dataloader(self, prefix="train", limit=None) -> DataLoader:
        """return {train/dev/test} dataloader"""
        #根据前缀（"train"、"dev"或"test"）获取相应数据加载器。

        dataset = self._load_dataset(prefix=prefix)

        if prefix == "train":
            batch_size = self.args.train_batch_size
            # small dataset like weibo ner, define data_generator will help experiment reproducibility.
            data_generator = torch.Generator()
            data_generator.manual_seed(self.args.seed)
            data_sampler = RandomSampler(dataset, generator=data_generator)
        else:
            batch_size = self.args.eval_batch_size
            data_sampler = SequentialSampler(dataset)

        # sampler option is mutually exclusive with shuffle
        dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size,
                                num_workers=self.args.workers, collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0]),
                                drop_last=False)

        return dataloader

    def test_dataloader(self, ) -> DataLoader:
        #: 获取测试数据加载器。

        return self.get_dataloader("test")

    def test_step(self, batch, batch_idx):
        #测试过程中的单步函数，用于计算测试评估指标。接收一个批次数据batch和批次索引batch_idx，返回测试评估指标。
        batch_images_list = batch[1]
        input_ids, pinyin_ids, gold_labels = batch[0]
        sequence_mask = (input_ids != 0).long()
        batch_size, seq_len = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, seq_len, 8)
        sequence_logits = self.forward(input_ids=input_ids, pinyin_ids=pinyin_ids,batch_images_list=batch_images_list)[0]
        probabilities, argmax_labels = self.postprocess_logits_to_labels(sequence_logits.view(batch_size, seq_len, -1))
        confusion_matrix = self.ner_evaluation_metric(argmax_labels, gold_labels, sequence_mask=sequence_mask)
        cws_confusion_matrix = self.cws_ner_evaluation_metric(argmax_labels, gold_labels, sequence_mask=sequence_mask)

        return {"confusion_matrix": confusion_matrix, "cws_confusion_matrix":cws_confusion_matrix}


    def test_epoch_end(self, outputs):
        #测试过程结束后的回调函数，用于计算测试评估指标。接收一个包含所有test_step返回值的列表outputs，返回一个包含测试评估指标的字典。

        confusion_matrix = torch.stack([x[f"confusion_matrix"] for x in outputs]).sum(0)
        all_true_positive, all_false_positive, all_false_negative = confusion_matrix
        cws_confusion_matrix = torch.stack([x[f"cws_confusion_matrix"] for x in outputs]).sum(0)
        cws_all_true_positive, cws_all_false_positive, cws_all_false_negative = cws_confusion_matrix

        if self.args.save_ner_prediction:
            precision, recall, f1, entity_tuple = self.ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative, prefix="test")
            cws_precision, cws_recall, cws_f1, cws_entity_tuple = self.cws_ner_evaluation_metric.compute_f1_using_confusion_matrix(cws_all_true_positive, cws_all_false_positive, cws_all_false_negative, prefix="test")
            gold_entity_lst, pred_entity_lst = entity_tuple
            self.save_predictions_to_file(gold_entity_lst, pred_entity_lst)
        else:
            precision, recall, f1 = self.ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative)

        tensorboard_logs = {"test_f1": f1,}
        self.result_logger.info(f"TEST RESULT -> TEST F1: {f1}, Precision: {precision}, Recall: {recall} ")
        tensorboard_logs = {"CWS_test_f1": cws_f1,}
        self.result_logger.info(f"CWS_TEST RESULT -> CWS_TEST F1: {cws_f1}, CWS_Precision: {cws_precision}, CWS_Recall: {cws_recall} ")


        print("=="*50)
        print("正在评价test上的结果：")
        print(f"TEST RESULT -> TEST F1: {f1}, Precision: {precision}, Recall: {recall} ")
        print(f"CWS_TEST RESULT -> CWS_TEST F1: {cws_f1}, CWS_Precision: {cws_precision}, CWS_Recall: {cws_recall} ")
        
        print("=="*50)
        return {"test_log": tensorboard_logs, "test_f1": f1, "test_precision": precision, "test_recall": recall}

    def postprocess_logits_to_labels(self, logits):
        # 后处理函数，将模型输出的logits转换为概率和预测标签。

        """input logits should in the shape [batch_size, seq_len, num_labels]"""
        bs, seqlen, num_labels = logits.shape
        mask = torch.zeros((bs, seqlen, num_labels)).to(logits.device)
        mask[:, :, 0] = -float('inf')
        logits = logits + mask        
        probabilities = F.softmax(logits, dim=2) # shape of [batch_size, seq_len, num_labels]
        argmax_labels = torch.argmax(probabilities, 2, keepdim=False) # shape of [batch_size, seq_len]
        return probabilities, argmax_labels

    def save_predictions_to_file(self, gold_entity_lst, pred_entity_lst, prefix="test"):
        # 将NER任务的预测结果保存到文件中，用于后续分析和查看。

        dataset = self._load_dataset(prefix=prefix)
        data_items = dataset.data_items
        if prefix=="test":
            file_path = "/public/home/dzhang/pyProject/hytian/ChineseBert_ADD_PIC_WEIBO/mertic/test_raw.txt"  # 请将文件路径替换为实际文件的路径
            save_file_path = os.path.join(self.args.save_path, "test_predictions.txt")
        else:
            file_path = "/public/home/dzhang/pyProject/hytian/ChineseBert_ADD_PIC_WEIBO/mertic/dev_raw.txt"
            save_file_path = os.path.join(self.args.save_path, "dev_predictions.txt")
        text_list = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                text_list.append(line.strip().split())  # 使用 strip() 去除每行末尾的换行符
        orig_text_list = []
        orig_gold_entity_lst = []
        for res in data_items:
            orig_text_list.append(res[0])
            orig_gold_entity_lst.append(res[1])

        print(f"INFO -> write predictions to {save_file_path}")
        with open(save_file_path, "w") as f:
            for gold_label_item, pred_label_item, text_list_item, orig_text_item, orig_gold_label_item in zip(gold_entity_lst, pred_entity_lst, text_list, orig_text_list, orig_gold_entity_lst):
                assert len(gold_label_item) == len(pred_label_item) == len(text_list_item)
                t = 0
                l = 0
                new_gold_label = []
                new_pred_label = []
                while t < len(orig_text_item):
                    ori_char = orig_text_item[t]
                    if ori_char == '년' or ori_char == '월' or ori_char == '일':
                        new_gold_label.append(orig_gold_label_item[t])
                        new_pred_label.append(orig_gold_label_item[t])
                        t+=1
                        l+=3
                        continue
                    if orig_text_item[t]=='️' and (orig_text_item[t-1] in ['❤','✈','♂','☕','✖']):
                        new_gold_label.append(orig_gold_label_item[t])
                        new_pred_label.append(orig_gold_label_item[t])
                        t+=1
                        continue
                    enc_char = text_list_item[l]
                    if ori_char.lower() == enc_char:
                        new_gold_label.append(gold_label_item[l])
                        new_pred_label.append(pred_label_item[l])
                        t+=1
                        l+=1
                    else:
                        if enc_char=="UNK":
                            temp_gold = gold_label_item[l]
                            temp_pred = pred_label_item[l]
                            if "S-" in temp_gold:
                                new_gold_label.append(temp_gold)
                                new_pred_label.append(temp_pred)
                                t+=1
                                l+=1
                            else:
                                if ori_char=='蝲':
                                    new_gold_label.append(temp_gold)
                                    new_pred_label.append(temp_pred)
                                    t+=1
                                    l+=1
                                    continue
                                n = t
                                while "E-" not in orig_gold_label_item[n]:
                                    n+=1
                                if "B-" in orig_gold_label_item[t]:
                                    new_gold_label.append("B-"+temp_gold.split("-")[1])
                                    for _ in range(t+1,n):
                                        new_gold_label.append("M-"+temp_gold.split("-")[1])
                                    new_gold_label.append("E-"+temp_gold.split("-")[1])
                                    
                                    new_pred_label.append("B-"+temp_pred.split("-")[1])
                                    for _ in range(t+1,n):
                                        new_pred_label.append("M-"+temp_pred.split("-")[1])
                                    new_pred_label.append("E-"+temp_pred.split("-")[1])
                                
                                elif "M-" in orig_gold_label_item[t]:
                                    for _ in range(t,n):
                                        new_gold_label.append("M-"+temp_gold.split("-")[1])
                                    new_gold_label.append("E-"+temp_gold.split("-")[1])  
                                    for _ in range(t,n):
                                        new_pred_label.append("M-"+temp_pred.split("-")[1])
                                    new_pred_label.append("E-"+temp_pred.split("-")[1])
                                else:
                                    new_gold_label.append("E-"+temp_gold.split("-")[1])  
                                    new_pred_label.append("E-"+temp_pred.split("-")[1])
                                                              
                                if ori_char=="ʕ":
                                    l += (n-t+1)
                                elif ori_char=="𝘁":
                                    l += 3
                                else:
                                    l += 1
                                t = n+1
                        else:
                            if "##" in enc_char:
                                enc_char = enc_char.replace("##","")
                            temp_gold = gold_label_item[l]
                            temp_pred = pred_label_item[l]
                            n = len(enc_char)
                            if n==1:
                                # if orig_text_item[t]=='️' and (orig_text_item[t-1] in ['❤','✈','♂']):
                                #     new_gold_label.append("E-"+orig_gold_label_item[t-1].split("-")[1])
                                #     new_pred_label.append("E-"+orig_gold_label_item[t-1].split("-")[1])
                                #     t+=1
                                # else:
                                new_gold_label.append(gold_label_item[l])
                                new_pred_label.append(pred_label_item[l])
                                t+=1
                                l+=1
                            else:
                                # if enc_char == "".join(orig_text_item[t:t+n]).lower():
                                if temp_gold.split("-")[0]=="B":
                                    new_gold_label.append("B-"+temp_gold.split("-")[1])
                                    for i in range(n-1):
                                        new_gold_label.append("M-"+temp_gold.split("-")[1])
                                elif temp_gold.split("-")[0]=="M":
                                    for i in range(n):
                                        new_gold_label.append("M-"+temp_gold.split("-")[1])
                                elif temp_gold.split("-")[0]=="E":
                                    for i in range(n-1):
                                        new_gold_label.append("M-"+temp_gold.split("-")[1])
                                    new_gold_label.append("E-"+temp_gold.split("-")[1])
                                else:
                                    new_gold_label.append("B-"+temp_gold.split("-")[1])
                                    for i in range(n-2):
                                        new_gold_label.append("M-"+temp_gold.split("-")[1])
                                    new_gold_label.append("E-"+temp_gold.split("-")[1])
                                
                                if temp_pred.split("-")[0]=="B":
                                    new_pred_label.append("B-"+temp_pred.split("-")[1])
                                    for i in range(n-1):
                                        new_pred_label.append("M-"+temp_pred.split("-")[1])
                                elif temp_pred.split("-")[0]=="M":
                                    for i in range(n):
                                        new_pred_label.append("M-"+temp_pred.split("-")[1])
                                elif temp_pred.split("-")[0]=="E":
                                    for i in range(n-1):
                                        new_pred_label.append("M-"+temp_pred.split("-")[1])
                                    new_pred_label.append("E-"+temp_pred.split("-")[1])
                                else:
                                    new_pred_label.append("B-"+temp_pred.split("-")[1])
                                    for i in range(n-2):
                                        new_pred_label.append("M-"+temp_pred.split("-")[1])
                                    new_pred_label.append("E-"+temp_pred.split("-")[1])
                                l+=1
                                t+=n        
                    assert new_gold_label==orig_gold_label_item[:t]
                    assert len(new_pred_label) == len(new_gold_label) == t
                
                assert new_gold_label == orig_gold_label_item
                assert len(new_gold_label) == len(new_pred_label) == len(orig_gold_label_item) == len(orig_text_item)
                f.write(" ".join(orig_text_item)+"\n")
                sentence=[]
                sentence1=[]
                gold_text_results=""
                pred_text_results=""
                for i in range(len(new_gold_label)):
                    if new_gold_label[i].split("-")[0]== "S" or new_gold_label[i].split("-")[0]=="E":
                        sentence.append(orig_text_item[i])
                        sentence.append(" ")
                    elif new_gold_label[i].split("-")[0]=="B" or new_gold_label[i].split("-")[0]=="M":
                        sentence.append(orig_text_item[i])
                gold_text_results="".join(sentence)
                for i in range(len(new_pred_label)):
                    if new_pred_label[i].split("-")[0]== "S" or new_pred_label[i].split("-")[0]=="E":
                        sentence1.append(orig_text_item[i])
                        sentence1.append(" ")
                    elif new_pred_label[i].split("-")[0]=="B" or new_pred_label[i].split("-")[0]=="M":
                        sentence1.append(orig_text_item[i])
                pred_text_results="".join(sentence1)              
                f.write("gold_text_results: "+gold_text_results+"\n")
                f.write("pred_text_results: "+pred_text_results+"\n")
                f.write("gold_label_item: "+" ".join(new_gold_label)+"\n")
                f.write("pred_label_item: "+" ".join(new_pred_label)+"\n")
                f.write("\n")

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_path", type=str, help="bert config file")
    parser.add_argument("--train_batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--use_memory", action="store_true", help="load dataset to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of dataset")
    parser.add_argument("--data_dir", type=str, help="train data path")
    parser.add_argument("--save_path", type=str, help="train data path")
    parser.add_argument("--save_topk", default=1, type=int, help="save topk checkpoint")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, )
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--classifier", type=str, default="single")
    parser.add_argument("--no_lr_scheduler", action="store_true")
    parser.add_argument("--save_ner_prediction", action="store_true", help="only work for test.")
    parser.add_argument("--path_to_model_hparams_file", default="", type=str, help="use for evaluation")
    parser.add_argument("--checkpoint_path", default="", type=str, help="use for evaluation.")
    parser.add_argument("--file_name", default="", type=str, help="use for truncated sets.")
    parser.add_argument("--task_name", default="ner", type=str, help="load different dataset")
    
    return parser


def main():
    parser = get_parser()
    #这里调用了一个名为get_parser()的函数，该函数返回一个argparse.ArgumentParser对象，用于解析命令行参数。

    parser = Trainer.add_argparse_args(parser)
    #将Trainer类中的参数添加到之前创建的argparse.ArgumentParser对象中。

    args = parser.parse_args()
    
    args_dict = vars(args)
    for i in args_dict:
        print("%s参数的值为：%s"%(i, str(args_dict[i])))
     #解析命令行参数，将它们保存在args变量中。

    model = WeiboTask(args)
    #创建一个名为WeiboTask的模型，并将解析后的参数args传递给该模型初始化函数，以便进行模型的配置和初始化。


    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_path, "checkpoint", "{epoch}",),
        #filepath: 表示保存模型检查点的文件路径。

        save_top_k=args.save_topk,
        #save_top_k: 表示要保存最好的几个模型检查点。

        save_last=False,
        #save_last: 表示是否保存最后一个（最后的epoch）模型检查点。在这里，False表示不保存最后一个检查点，即只保存save_top_k个最好的检查点。

        monitor="val_f1",
        #monitor: 表示要监视的指标名称，用于决定何时保存模型检查点。在这里，"val_f1"表示模型将会在验证集上的val_f1指标上进行监视，以便确定是否保存检查点。

        mode="max",
        #mode: 表示模型检查点保存的方式。在这里，"max"表示要保存具有最大val_f1值的检查点。

        verbose=True,
        #verbose: 表示是否在保存检查点时输出一些额外的信息。在这里，True表示在保存检查点时输出信息。

        period=-1,
        #period: 表示保存检查点的间隔，即每隔多少个epoch保存一次检查点。在这里，-1表示每个epoch都会保存检查点。

    )
    #创建一个ModelCheckpoint回调函数，该函数用于在训练过程中定期保存模型检查点。检查点会根据验证集上的val_f1指标来保存，只保存达到最高f1值的检查点。

    logger = TensorBoardLogger(
        save_dir=args.save_path,
        name='log')

    # save args
    with open(os.path.join(args.save_path, "checkpoint", "args.json"), "w") as f:
        #这里使用os.path.join()函数将文件路径拼接起来。

        args_dict = args.__dict__
        #这里使用args.__dict__将命令行参数args转换成一个字典类型的对象args_dict。

        del args_dict["tpu_cores"]
        #这里删除了args_dict字典中的一个键值对，即"tpu_cores"。

        json.dump(args_dict, f, indent=4)
        #使用json.dump()将args_dict字典的内容以JSON格式写入文件f中。indent=4参数是为了使得写入的JSON文件更加易读，它表示使用4个空格来缩进JSON数据。
    #这段代码的作用是将解析后的命令行参数保存为JSON格式的文件，以便后续跟踪和查看训练时使用的参数配置。

    trainer = Trainer.from_argparse_args(args,
                                         #args包含了模型训练过程中的各种参数，如学习率、批量大小、训练轮数等等。

                                         checkpoint_callback=checkpoint_callback,
                                         #这是一个关键字参数，指定了之前创建的ModelCheckpoint回调函数checkpoint_callback。

                                         logger=logger,
                                         #logger=logger: 这是另一个关键字参数，指定了之前创建的TensorBoardLogger对象logger。TensorBoardLogger用于记录训练过程中的日志信息

                                         deterministic=True)
                                        #deterministic=True: 这是一个关键字参数，用于设置是否使用随机数种子使得训练过程具有确定性。如果设置为True，则训练过程中使用的随机数种子将被固定，以便于结果的复现。
    #这段代码使用PyTorch Lightning库中的Trainer.from_argparse_args()方法来创建一个Trainer对象，用于管理深度学习模型的训练过程。

    trainer.fit(model)
    #使用Trainer对象对模型进行训练，这里model是之前创建的WeiboTask模型。

    # after training, use the model checkpoint which achieves the best f1 score on dev set to compute the f1 on test set.
    best_f1_on_dev, path_to_best_checkpoint = find_best_checkpoint_on_dev(args.save_path)
    #找到在开发集上获得最佳f1分数的检查点，这是为了在测试集上使用表现最好的模型。

    model.result_logger.info("=&"*20)
    #这行代码通过result_logger对象记录一条日志信息。这里的日志信息是一个包含20个"=&"的字符串，重复20次。

    print("=="*50)

    model.result_logger.info(f"Best F1 on DEV is {best_f1_on_dev}")
    #这行代码通过result_logger对象记录一条日志信息，输出开发集上的最佳F1值。

    print(f"Best F1 on DEV is {best_f1_on_dev}")

    model.result_logger.info(f"Best checkpoint on DEV set is {path_to_best_checkpoint}")
    #: 这行代码通过result_logger对象记录一条日志信息，输出开发集上的最佳模型检查点的路径。

    print(f"Best checkpoint on DEV set is {path_to_best_checkpoint}")
    checkpoint = torch.load(path_to_best_checkpoint)
    #加载开发集上表现最好的模型检查点。

    model.load_state_dict(checkpoint['state_dict'])
    #将模型的参数加载为开发集上表现最好的参数。

    trainer.test(model)
    #使用测试集对模型进行测试，评估模型在测试集上的性能。

    model.result_logger.info("=&"*20)
    print("=="*50)


def find_best_checkpoint_on_dev(output_dir: str, log_file: str = "eval_result_log.txt"):
    with open(os.path.join(output_dir, log_file)) as f:
        log_lines = f.readlines()

    F1_PATTERN=re.compile(r"val_f1 reached \d+\.\d* \(best")
    # val_f1 reached 0.00000 (best 0.00000)
    CKPT_PATTERN=re.compile(r"saving model to \S+ as top")
    checkpoint_info_lines = []
    for log_line in log_lines:
        if "saving model to" in log_line:
            checkpoint_info_lines.append(log_line)
    # example of log line
    # Epoch 00000: val_f1 reached 0.00000 (best 0.00000), saving model to /data/xiaoya/outputs/glyce/0117/debug_5_12_2e-5_0.001_0.001_275_0.1_1_0.25/checkpoint/epoch=0.ckpt as top 20
    best_f1_on_dev = 0
    best_checkpoint_on_dev = 0
    for checkpoint_info_line in checkpoint_info_lines:
        current_f1 = float(re.findall(F1_PATTERN, checkpoint_info_line)[0].replace("val_f1 reached ", "").replace(" (best", ""))
        current_ckpt = re.findall(CKPT_PATTERN, checkpoint_info_line)[0].replace("saving model to ", "").replace(" as top", "")

        if current_f1 >= best_f1_on_dev:
            best_f1_on_dev = current_f1
            best_checkpoint_on_dev = current_ckpt

    return best_f1_on_dev, best_checkpoint_on_dev


def evaluate():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = WeiboTask.load_from_checkpoint(checkpoint_path=args.checkpoint_path,
                                                            hparams_file=args.path_to_model_hparams_file,
                                                            map_location=None,
                                                            batch_size=1)
    trainer = Trainer.from_argparse_args(args, deterministic=True)

    trainer.test(model)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()


