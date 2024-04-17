# Introduction

this project is a guideline and relatived tools for pretraining LLM.

# Usage

## install prerequisite packages

```shell
python3 -m pip install torch flash-attn
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

## pretrain LlaMA2 7b

download pretrained llama2 7b checkpoint

```shell
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
cd Llama-2-7b-hf
git lfs pull
```

convert huggingface checkpoint to megatron checkpoint

```shell
cd <path/to/Megatron-LLaMA/root>
python3 tools/checkpoint_conversion/llama_checkpoint_conversion.py \
--load_path <path/to/huggingface/llama2/checkpoint> \
--save_path <path/to/megatron/llama2/checkpoint> \
--target_tensor_model_parallel_size 2 \
--target_pipeline_model_parallel_size 1 \
--target_data_parallel_size 16 \
--target_params_dtype "fp16" \
--make_vocab_size_divisible_by 1 \
--print-checkpoint-structure \
--megatron-path <path/to/Megatron-LLaMA/root>
```

pretrain on existing checkpoint

```shell
bash pretrain.sh
```

convert megatron checkpoint to huggingface checkpoint

```shell
cd <path/to/Megatron-LLaMA/root>
python3 tools/checkpoint_conversion/llama_checkpoint_conversion.py \
--convert_checkpoint_from_megatron_to_transformers \
--load_path <path/to/megatron/llama2/checkpoint> \
--save_path <path/to/huggingface/llama2/checkpoint> \
--target_params_dtype "fp16" \
--make_vocab_size_divisible_by 1 \
--print-checkpoint-structure \
--megatron-path <path/to/Megatron-LLaMA/root>
```

## supervised finetuning


