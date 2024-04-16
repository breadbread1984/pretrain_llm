# Introduction

this project is a guideline and relatived tools for pretraining LLM.

# Usage

## download and preprocess ebooks and papers

add query key words into a file. one line per query.

query and download the papers with the following command

```shell
python3 download_arxiv.py --input_txt <path/to/query/key/words> --output_dir <directory/to/output/paperas>
```

