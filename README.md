# Introduction

this project is a guideline and relatived tools for pretraining LLM.

# Usage

## install prerequisite packages

```shell
python3 -m pip install flash-attn
```

## download and preprocess ebooks and papers

add query key words into a file. one line per query. take **query_words.txt** as an example.

query and download the papers with the following command

```shell
python3 download_arxiv.py --input_txt <path/to/query/key/words> --output_dir <directory/to/output/paperas>
```

generate dataset from papers with the following command

```shell
python3 create_dataset.py --input_dir <directory/to/papers> --output_json <path/to/json>
```

download [Megatron-LlaMA](https://github.com/alibaba/Megatron-LLaMA) and preprocess the generated dataset with [**preprocess_data.py**](https://github.com/alibaba/Megatron-LLaMA/blob/main/tools/preprocess_data.py).

```shell
cd <path/to/Megatron-LLaMA/root>
python3 tools/preprocess_data.py --input <path/to/json> --output-prefix <prefix/for/output/bin/file> --tokenizer-type PretrainedFromHF --tokenizer-name-or-path <huggingface/model/id> --split-sentences
```

## pretrain LlaMA

```shell
cd <path/to/Megatron-LLaMA/root>
torchrun \
--nproc_per_node <gpu/per/node> \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6000 \
pretrain_llama.py \
--num-layers 32 \
--hidden-size 4096 \
--num-attention-heads 32 \
--seq-length 4096 \
--max-position-embeddings 4096 \
--micro-batch-size 4 \
--global-batch-size 8 \
--lr 0.00015 \
--train-iters 500000 \
--lr-decay-iters 320000 \
--lr-decay-style cosine \
--min-lr 1.0e-5 \
--weight-decay 1e-2 \
--lr-warmup-fraction .01 \
--clip-grad 1.0 \
--use-flash-attn \
--fp16 \
--data-path <prefix/for/dataset/files> \
--log-interval 100 \
--save-interval 10000 \
--eval-interval 1000 \
--eval-iters 10 \
--save llama2_ckpt \
--load llama2_ckpt \
--distributed-backend nccl \
--tensor-model-parallel-size <tensor/parallism/number> \
--pipeline-model-parallel-size <pipeline/parallism/number> \
--sequence-parallel
```

**NOTE**: <gpu/per/node> must equal to <tensor/parallism/number> * <pipeline/parallism/number>
