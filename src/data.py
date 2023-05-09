#!/usr/bin/python
# -*- encoding: utf-8 -*-

import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import os
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForWholeWordMask
from tqdm import tqdm
from transformers.utils import logging
logger = logging.get_logger(__name__)
import jieba

@dataclass
class SAILER_Collator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    encoder_mask_ratio: float = 0.15
    decoder_mask_ratio: float = 0.15

    def __post_init__(self):
        super().__post_init__()
        self.rng = random.Random()

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone() 

        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert

        probability_matrix = mask_labels  

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()  
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)  
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)   
            probability_matrix.masked_fill_(padding_mask, value=0.0)  

        masked_indices = probability_matrix.bool()  
        labels[~masked_indices] = -100  # We only compute loss on masked tokens 

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512, mlm_probability=0.15):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])


        self.rng.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
        
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms) 
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))] 
        return mask_labels

    def _truncate(self, example: List[int]):
        tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)
        if len(example) <= tgt_len:
            return example
        trunc = len(example) - tgt_len
        trunc_left = self.rng.randint(0, trunc)
        trunc_right = trunc - trunc_left

        truncated = example[trunc_left:]
        if trunc_right > 0:
            truncated = truncated[:-trunc_right]

        if not len(truncated) == tgt_len:
            print(len(example), len(truncated), trunc_left, trunc_right, tgt_len, flush=True)
            raise ValueError
        return truncated

    def _pad(self, seq, val=0):
        tgt_len = self.max_seq_length
        assert len(seq) <= tgt_len
        return seq + [val for _ in range(tgt_len - len(seq))]
    
    def encode_batch_examples_fact(self, examples, mlm_prob: float=0.15):
        encoded_examples = []
        masks = []
        mlm_masks = []

        for e in examples:    
        
            e_trunc = self._truncate(e['fact'])
            seg = jieba.lcut(e_trunc)
            tokens = []
            index = 1
            for i, word in enumerate(seg):
                if i>=0:
                    k = 0
                    for aa in word:
                        if(k > 0):
                            tokens.append("##"+aa)
                        else:
                            tokens.append(aa)
                        k = k + 1
            mlm_mask = self._whole_word_mask(tokens, mlm_probability=mlm_prob)
     
            mlm_mask = self._pad([0] + mlm_mask)  
         
            mlm_masks.append(mlm_mask)

            encoded = self.tokenizer.encode_plus(
                self._truncate(e['fact']),
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
            )
            masks.append(encoded['attention_mask'])
            encoded_examples.append(encoded['input_ids'])

        inputs, labels = self.mask_tokens(
            torch.tensor(encoded_examples, dtype=torch.long),
            torch.tensor(mlm_masks, dtype=torch.long)
        )
        attention_mask = torch.tensor(masks)

        batch = {
            "input_ids": inputs,
            "labels": labels,
            "attention_mask": attention_mask,
        }  

        return batch


    def encode_batch_examples_reason(self, examples, mlm_prob: float=0.15):
        encoded_examples = []
        masks = []
        mlm_masks = []

        for e in examples:    
            e_trunc = self._truncate(e['reason'])
            seg = jieba.lcut(e_trunc)
            tokens = []
            index = 1
            for i, word in enumerate(seg):
                if i>=0:
                    k = 0
                    for aa in word:

                        if(k > 0):
                            tokens.append("##"+aa)
                        else:
                            tokens.append(aa)
                        k = k + 1

            mlm_mask = self._whole_word_mask(tokens, mlm_probability=mlm_prob)
            
            mask = [0] +  mlm_mask
            
            mlm_mask = self._pad(mask[:512])
            mlm_masks.append(mlm_mask)

            encoded = self.tokenizer.encode_plus(
                e_trunc,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
            )
            masks.append(encoded['attention_mask'])
            encoded_examples.append(encoded['input_ids'])

        inputs, labels = self.mask_tokens(
            torch.tensor(encoded_examples, dtype=torch.long),
            torch.tensor(mlm_masks, dtype=torch.long)
        )
        attention_mask = torch.tensor(masks)

        batch = {
            "input_ids": inputs,
            "labels": labels,
            "attention_mask": attention_mask,
        }  

        return batch

    def encode_batch_examples_judgment(self, examples):
        encoded_examples = []
        masks = []
        mlm_masks = []

        for e in examples:    
            e_trunc = self._truncate(e['judgment'])
            
            cand_indexes = []
            tokens = [tid for tid in e_trunc]
            for (i, token) in enumerate(tokens):
                # print(token)
                cand_indexes.append(i)
                if token == '犯':
                    break

            mlm_mask = [1 if i in cand_indexes else 0 for i in range(len(tokens))] 

            e_trunc_law = self._truncate(e['law'])
            cand_indexes = []
            tokens = [tid for tid in e_trunc_law]
            for (i, token) in enumerate(tokens):
                if i > 6:
                    cand_indexes.append(i)
       
            mlm_mask_law = [1 if i in cand_indexes else 0 for i in range(len(tokens))] 

            mlm_mask = [0] + mlm_mask_law + mlm_mask
            mlm_mask = self._pad(mlm_mask[:512])
            mlm_masks.append(mlm_mask)

            encoded = self.tokenizer.encode_plus(
                e_trunc_law+e_trunc,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
            )
            masks.append(encoded['attention_mask'])
            encoded_examples.append(encoded['input_ids'])

        inputs, labels = self.mask_tokens(
            torch.tensor(encoded_examples, dtype=torch.long),
            torch.tensor(mlm_masks, dtype=torch.long)
        )
        attention_mask = torch.tensor(masks)

        batch = {
            "input_ids": inputs,
            "labels": labels,
            "attention_mask": attention_mask, 
        }  

        return batch

    def process(self, examples, cltypes=None):
        batch = self.encode_batch_examples_fact(examples=examples, mlm_prob=self.encoder_mask_ratio)  
        reason_batch = self.encode_batch_examples_reason(examples=examples, mlm_prob=self.decoder_mask_ratio)
        judgment_batch = self.encode_batch_examples_judgment(examples=examples)
       
        batch['reason_input_ids'] = reason_batch['input_ids']
        batch['reason_labels'] = reason_batch['labels']
        batch['reason_attention_mask'] = reason_batch['attention_mask']
        batch['judgment_input_ids'] = judgment_batch['input_ids']
        batch['judgment_labels'] = judgment_batch['labels']
        batch['judgment_attention_mask'] = judgment_batch['attention_mask']

        return batch
    
    def __call__(self, examples):
        unpacked = []
        for text_dict in examples:
            law = "根据《刑法》的" + text_dict['articles'] + '。'
            unpacked.append({'fact': text_dict['fact'], 'reason':text_dict['interpretation'], 'law':law, 'judgment':text_dict['judgment']})
        return self.process(unpacked)


class SAILER_Dataset(Dataset):
    def __init__(self, data_path, data_args):
        self.dataset = []
        f = open(data_path, "r", encoding="utf8")
        for line in tqdm(f):
            self.dataset.append(json.loads(line))
        self.data_args = data_args
        self.rng = random.Random()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):  
        return self.dataset[item]
