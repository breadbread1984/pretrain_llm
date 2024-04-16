# Introduction

this project is a guideline and relatived tools for pretraining LLM.

# Usage

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

