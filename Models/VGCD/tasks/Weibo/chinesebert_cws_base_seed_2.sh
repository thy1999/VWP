#!/usr/bin/env bash
# -*- coding: utf-8 -*-

TIME=1114
FILE_NAME=cws.addpics.char.bmes
FILE_TASK_NAME=weibo_cws_base_add_pic
REPO_PATH=/public/home/dzhang/pyProject/hytian/ChineseBert_ADD_PIC_WEIBO/ChineseBert-add-pic-cross-attention-after-encoder-weibo
BERT_PATH=/public/home/dzhang/pyProject/hytian/ChineseBert_ADD_PIC_WEIBO/pretrain_models/ChineseBERT-base
DATA_DIR=/public/home/dzhang/pyProject/hytian/ChineseBert_ADD_PIC_WEIBO/ChineseBert-add-pic-cross-attention-after-encoder-weibo/data/weibo_split_long_cws_add_pics

TASK_NAME=cws
SAVE_TOPK=1
TRAIN_BATCH_SIZE=64
LR=2e-5
WEIGHT_DECAY=0.01
WARMUP_PROPORTION=0.02
MAX_LEN=512
MAX_EPOCH=10
DROPOUT=0.1
ACC_GRAD=1
VAL_CHECK_INTERVAL=0.5
OPTIMIZER=torch.adam
CLASSIFIER=multi
SEED=2

OUTPUT_DIR=/public/home/dzhang/pyProject/hytian/ChineseBert_ADD_PIC_WEIBO/ChineseBert-add-pic-cross-attention-after-encoder-weibo/exp/${TIME}/${FILE_TASK_NAME}_${MAX_EPOCH}_${SEED}_${TRAIN_BATCH_SIZE}_${LR}_${WEIGHT_DECAY}_${WARMUP_PROPORTION}_${MAX_LEN}_${DROPOUT}_${ACC_GRAD}_${VAL_CHECK_INTERVAL}
mkdir -p $OUTPUT_DIR
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

# CUDA_VISIBLE_DEVICES=0
python3 $REPO_PATH/tasks/Weibo/Weibo_trainer.py \
--lr ${LR} \
--max_epochs ${MAX_EPOCH} \
--max_length ${MAX_LEN} \
--weight_decay ${WEIGHT_DECAY} \
--hidden_dropout_prob ${DROPOUT} \
--warmup_proportion ${WARMUP_PROPORTION}  \
--train_batch_size ${TRAIN_BATCH_SIZE} \
--accumulate_grad_batches ${ACC_GRAD} \
--save_topk ${SAVE_TOPK} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--classifier ${CLASSIFIER} \
--gpus="1" \
--optimizer ${OPTIMIZER} \
--bert_path ${BERT_PATH} \
--data_dir ${DATA_DIR} \
--save_path ${OUTPUT_DIR} \
--file_name ${FILE_NAME} \
--task_name ${TASK_NAME} \
--seed ${SEED} \
--save_ner_prediction


