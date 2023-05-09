#!/bin/bash
###
 # @Author: lihaitao
 # @Date: 2023-04-28 14:32:33
 # @LastEditors: Do not edit
 # @LastEditTime: 2023-04-28 14:33:47
 # @FilePath: /lht/GitHub_code/SAILER/src/sailer.sh
### 

# Model name & Output Path
MODEL_NAME=${0%.*}   # Use filename as model's output dir name
OUTPUT_DIR=results/$MODEL_NAME

if [ ! -d $OUTPUT_DIR/model ]; then
  mkdir -p $OUTPUT_DIR/model
  echo "makedir $OUTPUT_DIR/model"
fi

if [ ! -d $OUTPUT_DIR/logs ]; then
  mkdir -p $OUTPUT_DIR/logs
  echo "makedir $OUTPUT_DIR/logs"
fi

if [ ! -d $OUTPUT_DIR/tfboard/$MODEL_NAME ]; then
  mkdir -p $OUTPUT_DIR/tfboard/$MODEL_NAME
  echo "makedir $OUTPUT_DIR/tfboard/$MODEL_NAME"
fi


BATCH_SIZE_PER_GPU=36
GRAD_ACCU=4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node 8 \
  --master_port 29508 \
  run_pretraining.py \
  --model_name_or_path bert-base-chinese \
  --output_dir $OUTPUT_DIR/model/SAILER_bert \
  --do_train \
  --logging_steps 50 \
  --save_steps 500 \
  --fp16 \
  --logging_dir $OUTPUT_DIR/tfboard/$MODEL_NAME \
  --warmup_ratio 0.1 \
  --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
  --gradient_accumulation_steps $GRAD_ACCU \
  --learning_rate 5e-5 \
  --overwrite_output_dir \
  --dataloader_drop_last \
  --dataloader_num_workers 4 \
  --max_seq_length 512 \
  --num_train_epochs 10 \
  --train_path ../data/data_example.json \
  --weight_decay 0.01 \
  --encoder_mask_ratio 0.15 \
  --decoder_mask_ratio 0.45 \
  --use_decoder_head \
  --enable_head_mlm \
  --ddp_find_unused_parameters False \
  --n_head_layers 1 \
  |& tee $OUTPUT_DIR/logs/run_pretraining.log
