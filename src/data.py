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
jieba.load_userdict('./dict.txt')


dic = {'1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九', '0': ''}
def numToCharater(word:str):
    length = len(word)
    if length == 3:
        if word[1] == '0':
            if word[2] == '0':
                return dic[word[0]] + '百'
            else:
                return dic[word[0]] + '百' + '零' + dic[word[2]]
        else:
            return dic[word[0]] + '百' + dic[word[1]] + '十' + dic[word[2]]
    elif length == 2:
        if word[0] == '1':
            return '十' + dic[word[1]]
        else:
            return dic[word[0]] + '十' + dic[word[1]]
    elif length == 1:
        return dic[word[0]]



@dataclass
class SAILER_Collator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    encoder_mask_ratio: float = 0.15
    decoder_mask_ratio: float = 0.15
    data_type: str = 'mixed'

    def __post_init__(self):
        super().__post_init__()
        self.rng = random.Random()

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        可以使用wwm试一下，如果可以mask法律要素就更好了
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone() ##先复制一下正确答案

        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels  ##这个代表着想mask哪些label

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()  ###返回一个【0，1，1，0】的list，代表是不是特殊的token
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)  ###用0把true地方填充
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)   ##与某个值比较，一样则返回true
            probability_matrix.masked_fill_(padding_mask, value=0.0)  ##如果是pad则填上0

        masked_indices = probability_matrix.bool()  ###哪些mask 哪些不mask
        labels[~masked_indices] = -100  # We only compute loss on masked tokens 是的

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

        # print(cand_indexes)
        self.rng.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
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

        assert len(covered_indexes) == len(masked_lms) ###哪些index被mask了 
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]  ##z遮挡的用1
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

        for e in examples:    ###每一条里面都有一个text文件
        
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
     
            mlm_mask = self._pad([0] + mlm_mask)  ###为什么加了一个0 对应cls吗
         
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
            # "input_ids_unmasked": torch.tensor(encoded_examples, dtype=torch.long),
        }  

        return batch


    def encode_batch_examples_reason(self, examples, mlm_prob: float=0.15):
        encoded_examples = []
        masks = []
        mlm_masks = []

        for e in examples:    ###每一条里面都有一个text文件

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
            
            cand_indexes = []
            tokens = [tid for tid in self._truncate(e['cause'])]
            for (i, token) in enumerate(tokens):
                # print(i)
                # print(token)
                if i > 6:
                    cand_indexes.append(i)
            mlm_mask_cause = [1 if i in cand_indexes else 0 for i in range(len(tokens))] 


            # cand_indexes = []
            # tokens = [tid for tid in self._truncate(e['law'])]
            # for (i, token) in enumerate(tokens):
            #     # print(i)
            #     # print(token)
            #     if i > 6:
            #         cand_indexes.append(i)
       
            # mlm_mask_law = [1 if i in cand_indexes else 0 for i in range(len(tokens))] 

            # mask = [0] + mlm_mask_cause + mlm_mask

            mask = [0] +  mlm_mask
            
            mlm_mask = self._pad(mask[:512])
            # print(mlm_mask)
            # print(e['law']+e['cause']+e['reason'])
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
            # "input_ids_unmasked": torch.tensor(encoded_examples, dtype=torch.long),
        }  ###处理的好复杂

        return batch

    def encode_batch_examples_cause(self, examples):
        encoded_examples = []
        masks = []
        mlm_masks = []

        for e in examples:    ###每一条里面都有一个text文件
            e_trunc = self._truncate(e['judgment'])
            
            # print(e_trunc)
            cand_indexes = []
            tokens = [tid for tid in e_trunc]
            for (i, token) in enumerate(tokens):
                # print(token)
                cand_indexes.append(i)
                if token == '犯':
                    break

       
            mlm_mask = [1 if i in cand_indexes else 0 for i in range(len(tokens))] 

            e_trunc_law = self._truncate(e['judgment'])
            cand_indexes = []
            tokens = [tid for tid in e_trunc_law]
            for (i, token) in enumerate(tokens):
                # print(i)
                # print(token)
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
            # "input_ids_unmasked": torch.tensor(encoded_examples, dtype=torch.long),
        }  ###处理的好复杂

        return batch

    def process(self, examples, cltypes=None):
        batch = self.encode_batch_examples_fact(examples=examples, mlm_prob=self.encoder_mask_ratio)  ###这个是你一条我一条
        decoder_batch = self.encode_batch_examples_reason(examples=examples, mlm_prob=self.decoder_mask_ratio) ###输出decoder的编码
        cause_batch = self.encode_batch_examples_cause(examples=examples)
       

        batch['decoder_input_ids'] = decoder_batch['input_ids']
        batch['decoder_labels'] = decoder_batch['labels']
        batch['decoder_attention_mask'] = decoder_batch['attention_mask']
        batch['cause_input_ids'] = cause_batch['input_ids']
        batch['cause_labels'] = cause_batch['labels']
        batch['cause_attention_mask'] = cause_batch['attention_mask']

        return batch
    
    def __call__(self, examples):
        unpacked = []
        for text_dict in examples:
            try:
                articles = ','.join(['第' + numToCharater(str(num)) + '条' for num in sorted(text_dict['meta']['relevant_articles'])])
            except:
                print(text_dict['meta']['relevant_articles'])
                articles = '第六百条'
            law = "根据《刑法》的" + articles + '。'
            # print(law)
            cause = "被告人被判为:" + ",".join([accu + '罪' for accu in set(text_dict['meta']['accusation'])]) + '。'
            unpacked.append({'fact': text_dict['fact'],'reason':text_dict['interpretation'],'cause':cause,'law':law, 'judgment':text_dict['judgment']})
           
            # unpacked.append({'text': text_dict['interpretation']})
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

    def __getitem__(self, item):   ###这个地方需要修改，返回的不是随机选取的
        return self.dataset[item]
