# CoT-MAE
CoT-MAE is a transformers based Mask Auto-Encoder pretraining architecture designed for Dense Passage Retrieval. Details can be found in [ConTextual Mask Auto-Encoder for Dense Passage Retrieval](https://arxiv.org/abs/2208.07670).

## Get Started
Checkpoints are available at Huggingface Hub.

| Model                         | Descriptions                                                                              | Link                                                                                                  |
|-------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| cotmae_base_uncased           | CoT-MAE Pretrained on unsupervised MS-Marco documents                                     | [caskcsg/cotmae_base_uncased](https://huggingface.co/caskcsg/cotmae_base_uncased)                     |
| cotmae_base_msmarco_retriever | CoT-MAE Retriever finetuned on MS-Marco with BM25 hard negatives and mined hard negatives | [caskcsg/cotmae_base_msmarco_retriever](https://huggingface.co/caskcsg/cotmae_base_msmarco_retriever) |
| cotmae_base_msmarco_reranker  | CoT-MAE Reranker finetuned on MS-Marco with mined hard negatives                          | [caskcsg/cotmae_base_msmarco_reranker](https://huggingface.co/caskcsg/cotmae_base_msmarco_reranker)   |


## Dependencies
Please refer to [PyTorch Homepage](https://pytorch.org/) to install a pytorch version suitable for your system.

Dependencies can be installed by running codes below. Specifically, we use transformers=4.17.0 for our experiments. Other versions should also work well.
```bash
apt-get install parallel
pip install transformers==4.17.0 datasets nltk tensorboard pandas tabulate
```

We use [Tevatron](https://github.com/texttron/tevatron) toolkit for finetuning. You can install it by following its guidelines.

## Pre-training
### Data processing
[MS-Marco documents](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz) are used as unsupervised pre-training corpus. You can download and process the data by running [get_pretraining_data.sh](./get_pretraining_data.sh) script. The processed contextual texts will be stored in **data/msmarco-docs.mlen128.json** as json format.
```bash
bash get_pretraining_data.sh
```

For MS-Marco data processing of Tevatron toolkit, just running script at [msmarco/get_data.sh](msmarco/get_data.sh). The processed data will be stored in **msmarco/marco** folder.
```bash
cd msmarco
bash get_data.sh
```

The below **pretraining and MS-Marco finetuning** pipeline are integrated in script [cotmae.sh](./cotmae.sh). You can have a try on it, or follow the codes below.

### Pre-training
The code below will launch pre-training on 8-gpus and train CoT-MAE warm start from bert-base-uncased.
```bash
TOTAL_STEPS=800000
TOTAL_BATCH_SIZE=1024
TOTAL_BATCH_SIZE=$((TOTAL_BATCH_SIZE/2))  # one text generates two spans: 'anchor', 'contextual_span'
BATCH_SIZE_PER_GPU=64
GRAD_ACCU=1
OUTPUT_DIR=results/cotmae

python -m torch.distributed.launch \
  --nproc_per_node 8 \
  run_pretraining.py \
  --model_name_or_path bert-base-uncased \
  --output_dir $OUTPUT_DIR/model \
  --do_train \
  --logging_steps 20 \
  --save_steps 100000 \
  --save_total_limit 4 \
  --fp16 \
  --logging_dir $OUTPUT_DIR/tfboard/cotmae \
  --warmup_ratio 0.1 \
  --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
  --gradient_accumulation_steps $GRAD_ACCU \
  --learning_rate 1e-4 \
  --max_steps $TOTAL_STEPS \
  --overwrite_output_dir \
  --dataloader_drop_last \
  --dataloader_num_workers 16 \
  --max_seq_length 128 \
  --train_path data/msmarco-docs.mlen128.json \
  --weight_decay 0.01 \
  --data_type mixed \
  --encoder_mask_ratio 0.30 \
  --decoder_mask_ratio 0.45 \
  --use_decoder_head \
  --enable_head_mlm \
  --n_head_layers 2

```
We use a much small learn rate **1e-4** & batch size **1k** with longer enough steps as described in our paper. It takes 2.5 days to finish 800k steps pre-training on **8 A100 gpus**. 

### Finetuning
#### Finetuning on MS-Marco Passage ranking task with Tevatron
[Tevatron](https://github.com/texttron/tevatron) toolkit for finetuning. Please refer to [msmarco/eval_msmarco.sh](./msmarco/eval_msmarco.sh) for more details. You can run a two stage finetuning by just running this scripts.
```bash
cd msmarco
# Assume the pre-trained model is located in ./results/cotmae/model
bash eval_msmarco.sh cotmae
```
Scores of CoT-MAE that pre-trained for 800k steps & 1200k steps are listed as follows. Typically a longer training steps will increase the performance.
| Models       | MRR @10  | recall@1 | recall@50 | recall@1k | QueriesRanked  |
|--------------|----------|----------|-----------|-----------|----------------|
| cotmae-800k  | 0.391029 | 0.260172 | 0.875072  | 0.987679  | 6980           |
| cotmae-1200k | 0.394431 | 0.265903 | 0.870344  | 0.986676  | 6980           |

## Citations
If you find our work useful, please cite our paper.
```bibtex
@misc{https://doi.org/10.48550/arxiv.2208.07670,
  doi = {10.48550/ARXIV.2208.07670},
  url = {https://arxiv.org/abs/2208.07670},
  author = {Wu, Xing and Ma, Guangyuan and Lin, Meng and Lin, Zijia and Wang, Zhongyuan and Hu, Songlin},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {ConTextual Mask Auto-Encoder for Dense Passage Retrieval},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
