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
from modeling import BertForSAILER
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
        # let's parse it to get our arguments. 
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()  


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)], 
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

    
    train_dataset = SAILER_Dataset(
    data_args.train_path
    ,data_args)
    eval_dataset = None

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.


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
        model = BertForSAILER.from_pretrained(
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
        model = BertForSAILER.from_config(
                        config,
                        use_decoder_head=model_args.use_decoder_head,
                        n_head_layers=model_args.n_head_layers,
                        enable_head_mlm=model_args.enable_head_mlm,
                        head_mlm_coef=model_args.head_mlm_coef,
                    )

    model.resize_token_embeddings(len(tokenizer))  ###Pointer to the input tokens Embeddings Module of the model.

    # # Data collator 
    data_collator = SAILER_Collator(
        tokenizer=tokenizer,
        encoder_mask_ratio=data_args.encoder_mask_ratio,
        decoder_mask_ratio=data_args.decoder_mask_ratio,
        max_seq_length=data_args.max_seq_length,
    )

    # Initialize our Trainer
    trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()  

if __name__ == "__main__":
    main()
