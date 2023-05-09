#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from typing import Optional

import datasets
from datasets import load_dataset
from torch.utils.data import SequentialSampler,RandomSampler
import torch
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import is_main_process

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from modeling import BertForCotMAE
from data import SAILER_Dataset, SAILER_Collator
from arguments import ModelArguments, DataTrainingArguments, SAILER_PreTrainingArguments as TrainingArguments
from trainer import TrainerWithLogs as Trainer

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
from torch.utils.data import DataLoader



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments. 如果是配置文件，那么直接加载配置文件
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()  ##获得参数


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)], #将日志消息写入已经打开的类文件对象fileobj
    )

    log_level = logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


    # Log on each process the small summary: 
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    # Set seed before initializing model.
    set_seed(training_args.seed)
 
    train_dict = {}

    
    train_dataset = CotMAEDataset(
    data_args.train_path
    ,data_args)
    eval_dataset = None

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir, use_fast=False
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=False
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = BertForCotMAE.from_pretrained(
                    pretrained_model_name_or_path=model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    use_decoder_head=model_args.use_decoder_head,
                    n_head_layers=model_args.n_head_layers,
                    enable_head_mlm=model_args.enable_head_mlm,
                    head_mlm_coef=model_args.head_mlm_coef,
                )
    else:
        logger.warning('Training from scratch.')
        model = BertForCotMAE.from_config(
                        config,
                        use_decoder_head=model_args.use_decoder_head,
                        n_head_layers=model_args.n_head_layers,
                        enable_head_mlm=model_args.enable_head_mlm,
                        head_mlm_coef=model_args.head_mlm_coef,
                    )

    model.resize_token_embeddings(len(tokenizer))  ###Pointer to the input tokens Embeddings Module of the model.
    # ##这句话重新设置了embedding

    # # Data collator  ###这个是重点
    data_collator = CotMAECollator(
        tokenizer=tokenizer,
        encoder_mask_ratio=data_args.encoder_mask_ratio,
        decoder_mask_ratio=data_args.decoder_mask_ratio,
        max_seq_length=data_args.max_seq_length,
        data_type=data_args.data_type,
    )

    # sampler = RandomSampler(train_dataset)    
    # dataloader = DataLoader(dataset=train_dataset,
    #                             batch_size=1,
    #                             num_workers=4,
    #                             collate_fn=data_collator,
    #                             drop_last=True,
    #                             sampler=sampler)

    # for step, data in enumerate(dataloader):
    #     print(data)
    #     break


    # Initialize our Trainer
    trainer = Trainer(  ###确实好用
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        # trainer.train(model_path=model_path)
        trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload   ##保存 只用encoder


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
