#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# author: xiaoya li
# Result:
# TEST F1: 0.7095990180969238, Precision: 0.7019230723381042, Recall: 0.7174447178840637

TIME=0529
FILE_NAME=char.bmes
FILE_TASK_NAME=weibo_pos_large
REPO_PATH=/public/home/dzhang/pyProject/hytian/ChineseBert-weibo/ChineseBert-main
BERT_PATH=/public/home/dzhang/pyProject/hytian/ChineseBert-weibo/ChineseBert-main/pretrain_model/ChineseBERT-large
DATA_DIR=/public/home/dzhang/pyProject/hytian/ChineseBert-weibo/ChineseBert-main/data/weibo_split_long_pos

TASK_NAME=pos
SAVE_TOPK=2
# need change
TRAIN_BATCH_SIZE=18
LR=2e-5
WEIGHT_DECAY=0.001
WARMUP_PROPORTION=0.02
MAX_LEN=512
MAX_EPOCH=10
DROPOUT=0.1
ACC_GRAD=1
VAL_CHECK_INTERVAL=0.25
OPTIMIZER=torch.adam
CLASSIFIER=multi

OUTPUT_DIR=/public/home/dzhang/pyProject/hytian/ChineseBert-weibo/ChineseBert-main/${TIME}/${FILE_TASK_NAME}_${MAX_EPOCH}_${TRAIN_BATCH_SIZE}_${LR}_${WEIGHT_DECAY}_${WARMUP_PROPORTION}_${MAX_LEN}_${DROPOUT}_${ACC_GRAD}_${VAL_CHECK_INTERVAL}
mkdir -p $OUTPUT_DIR
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
# CUDA_VISIBLE_DEVICES=1 
python3 $REPO_PATH/tasks/Weibo/Weibo_trainer.py \
--lr ${LR} \
--max_epochs ${MAX_EPOCH} \
--max_length ${MAX_LEN} \
--weight_decay ${WEIGHT_DECAY} \
--hidden_dropout_prob ${DROPOUT} \
--warmup_proportion ${WARMUP_PROPORTION} \
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
--task_name ${TASK_NAME}

