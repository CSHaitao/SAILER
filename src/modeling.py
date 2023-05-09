#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Model Implementation

@Author  :   Ma (Ma787639046@outlook.com)
'''
from typing import Optional, Dict
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import BertForMaskedLM
from transformers.models.bert.modeling_bert import BertLayer
from transformers.modeling_outputs import MaskedLMOutput

@dataclass
class MaskedLMOutputWithLogs(MaskedLMOutput):
    logs: Optional[Dict[str, any]] = None

class BertForSAILER(BertForMaskedLM):
    def __init__(
        self,
        config,
        use_decoder_head: bool = True,
        n_head_layers: int = 2,
        enable_head_mlm: bool = True,
        head_mlm_coef: float = 1.0,
    ):
        super().__init__(config)
        if use_decoder_head:
            self.c_head = nn.ModuleList(
                [BertLayer(config) for _ in range(n_head_layers)]
            )     ####两层bertlayer
            self.c_head.apply(self._init_weights)  ##初始化

        self.cause_head = nn.ModuleList(
                [BertLayer(config) for _ in range(1)]
            )     ####两层bertlayer
        self.cause_head.apply(self._init_weights)  ##初始化


        self.cross_entropy = nn.CrossEntropyLoss()

        self.use_decoder_head = use_decoder_head
        self.n_head_layers = n_head_layers
        self.enable_head_mlm = enable_head_mlm
        self.head_mlm_coef = head_mlm_coef

    def forward(self, **model_input):
        lm_out: MaskedLMOutput = super().forward(
            input_ids = model_input['input_ids'],
            attention_mask = model_input['attention_mask'],
            labels=model_input['labels'],
            output_hidden_states=True,
            return_dict=True
        )  ###注意输入的形式 感觉super是BertForMaskLM的输出结果

        cls_hiddens = lm_out.hidden_states[-1][:, 0] ###最后一层[:,0]所有行第0列  代表最后输出的cls向量

        logs = dict()

        # add last layer mlm loss
        loss = lm_out.loss
        logs["encoder_mlm_loss"] = lm_out.loss.item() ###encoder的loss
        
        if self.use_decoder_head and self.enable_head_mlm:
            # Get the embedding of decoder inputs
            decoder_embedding_output = self.bert.embeddings(input_ids=model_input['decoder_input_ids']) ###获取embeddings
            decoder_attention_mask = self.get_extended_attention_mask(
                                        model_input['decoder_attention_mask'],
                                        model_input['decoder_attention_mask'].shape,
                                        model_input['decoder_attention_mask'].device
                                    )
           
            # Concat cls-hiddens of span A & embedding of span B
            ## cls输出的向量和decoder输入的向量拼起来
            hiddens = torch.cat([cls_hiddens.unsqueeze(1), decoder_embedding_output[:, 1:]], dim=1)
            for layer in self.c_head:
                layer_out = layer(
                    hiddens,
                    decoder_attention_mask,
                )
                hiddens = layer_out[0]
                # print("222")
                # print(cls_hiddens.unsqueeze(1).shape)
                # print("333")
                # print (hiddens.shape)
                reason_embedding = hiddens[:,0,:]
                # print("44")
                # print(reason_embedding.shape)
                
                # print(model_input['decoder_labels'])
            # add head-layer mlm loss

            head_mlm_loss = self.mlm_loss(hiddens, model_input['decoder_labels']) * self.head_mlm_coef ###decoder的loss比例
            logs["head_mlm_loss"] = head_mlm_loss.item()



            decoder_embedding_output = self.bert.embeddings(input_ids=model_input['cause_input_ids']) ###获取embeddings
            decoder_attention_mask = self.get_extended_attention_mask(
                                        model_input['cause_attention_mask'],
                                        model_input['cause_attention_mask'].shape,
                                        model_input['cause_attention_mask'].device
                                    )
            # Concat cls-hiddens of span A & embedding of span B
            # cls输出的向量和decoder输入的向量拼起来
            # hiddens = torch.cat([reason_embedding.unsqueeze(1), decoder_embedding_output[:, 1:]], dim=1)
            hiddens = torch.cat([cls_hiddens.unsqueeze(1), decoder_embedding_output[:, 1:]], dim=1)
            # new_embedding = torch.cat([cls_hiddens.unsqueeze(1), reason_embedding.unsqueeze(1)], dim=1)
            # hiddens = torch.cat([new_embedding, decoder_embedding_output[:, 1:-1]], dim=1)
            for layer in self.cause_head:
                layer_out = layer(
                    hiddens,
                    decoder_attention_mask,
                )
                hiddens = layer_out[0]
            # add head-layer mlm loss

            cause_mlm_loss = self.mlm_loss(hiddens, model_input['cause_labels']) * self.head_mlm_coef ###decoder的loss比例
            logs["cause_mlm_loss"] = cause_mlm_loss.item()
            
            loss += head_mlm_loss
            loss += cause_mlm_loss
          

        
        return MaskedLMOutputWithLogs(
            loss=loss,
            logits=lm_out.logits,  #
            hidden_states=lm_out.hidden_states,
            attentions=lm_out.attentions,
            logs=logs,   
        )

    def mlm_loss(self, hiddens, labels):
        pred_scores = self.cls(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.config.vocab_size),
            labels.view(-1)
        )
        return masked_lm_loss
