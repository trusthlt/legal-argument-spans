# Contemporary LLMs struggle with extracting formal legal arguments

Companion code to the [NLLP 2025 paper](https://aclanthology.org/2025.nllp-1.20/) *"Contemporary LLMs struggle with extracting formal legal arguments"*.

> Legal Argument Mining (LAM) is a complex challenge for humans and language models alike. This paper explores the application of Large Language Models (LLMs) in LAM, focusing on the identification of fine-grained argument types within judgment texts. We compare the performance of Flan-T5 and Llama 3 models against a baseline RoBERTa model to study if the advantages of magnitude-bigger LLMs can be leveraged for this task. Although the tested models succeed at implementing the task in a generative fashion, our findings indicate that neither fine-tuning nor prompting could surpass the performance of a domain-pre-trained encoder-only model.

---

## Table of Contents
- [Installation](#installation)
- [Repository structure](#repository-structure)
- [Generating datasets](#datasets)
- [Running experiments](#experiments)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Contact](#contact)

---

## Installation
Clone the repo, create a Conda environment, and install dependencies:
```bash
git clone https://github.com/trusthlt/legal-argument-spans.git
cd legal-argument-spans
```
```bash
conda create --name spans python=3.11
conda activate spans
pip install -r requirements.txt
```
All experiments were run on a single NVIDIA A100 80GB GPU.

## Repository structure
| File | Description |
| ---- | ----------- |
| `config.ini` | Paths to the gold data, the prediction CSVs scored by `03`, and the shared annotation tokens. |
| `00_extract_data_from_xmi.py` | Extract paragraphs + annotations from the LAM:ECHR XMI gold data and reformat the BIO tags into in-text span annotations (`data/preprocessed_dataset.json`). |
| `01_create_windowed_dataset.py` | Build the context-window variants (0, 2, 4 previous paragraphs) and the full-document dataset as Hugging Face `DatasetDict`s. |
| `02A_train_t5.py` | Fine-tune (LoRA) and run inference with Flan-T5. |
| `02B_train_llama.py` | Fine-tune (QLoRA) / ICL and run inference with the Llama 3 models. |
| `02C_transform_lam_baseline.py` | Transform the original RoBERTa **BIO** baseline predictions into the in-text annotation format and write the baseline CSV read by `03`. |
| `03_evaluate_predictions.py` | Score the prediction CSVs (ACI, ACC, output-format validity) with the relaxed span metric. |
| `utils.py` | Shared helpers: label mapping, span parsing, annotation guidelines. |

## Datasets

The datasets are based on the LAM:ECHR dataset ([trusthlt/mining-legal-arguments](https://github.com/trusthlt/mining-legal-arguments)) and the original BIO tags are reformatted into in-text annotations.

The LAM:ECHR gold data and the original RoBERTa baseline predictions are **not shipped** with this repository. You can copy the dataset into the folders using:
```bash
curl -L -o mining-legal-arguments.zip https://github.com/trusthlt/mining-legal-arguments/archive/refs/heads/main.zip
unzip -q mining-legal-arguments.zip "mining-legal-arguments-main/gold_data/*" "mining-legal-arguments-main/predictions/roberta-large-fp-15000_checkpoint-143334_roberta-argType.csv"
mv mining-legal-arguments-main/gold_data/* lam_gold_data/
mv mining-legal-arguments-main/predictions/roberta-large-fp-15000_checkpoint-143334_roberta-argType.csv lam_predictions/original_predictions_roberta_fp_15k.csv
rm -rf mining-legal-arguments.zip mining-legal-arguments-main
```

Generate the reformatted data and the context-window variants:
```bash
# reformat the gold data into in-text annotations -> data/preprocessed_dataset.json
conda run -n spans python 00_extract_data_from_xmi.py
# create the datasets with context windows 0 (= no context), 2 and 4, and full document input
conda run -n spans python 01_create_windowed_dataset.py --input data/preprocessed_dataset.json --output_dir data --window 0
conda run -n spans python 01_create_windowed_dataset.py --input data/preprocessed_dataset.json --output_dir data --window 2
conda run -n spans python 01_create_windowed_dataset.py --input data/preprocessed_dataset.json --output_dir data --window 4
conda run -n spans python 01_create_windowed_dataset.py --input data/preprocessed_dataset.json --output_dir data --full
```

## Experiments

To replicate the experiments in the paper, run the scripts with the parameters listed in the table below.

| Configuration | Model | Version | Epochs | Batch size | Gradient acc. steps | Max seq. | Lora rank | Lora alpha | lr |
| ------------- | ----- | ------- | ------ | ---------- | ------------------- | -------- | --------- | ---------- | --- |
| Flan-T5-XXL context = 0 | Flan-T5 XXL | - | 2 | 16 | - | 512 | 16 | 32 | 1e-3 |
| Llama 3.1 8B ICL | unsloth/Meta-Llama-3.1-8B-Instruct | 1 | 1 | 4 | 8 | 4000 | 16 | 32 | 2e-4 |
| Llama 3.1 8B context = 0 | unsloth/Meta-Llama-3.1-8B-Instruct | 0 | 1 | 4 | 8 | 4000 | 16 | 32 | 2e-4 |
| Llama 3.1 8B context = 2 | unsloth/Meta-Llama-3.1-8B-Instruct | 2 | 1 | 4 | 8 | 4000 | 16 | 32 | 2e-4 |
| Llama 3.1 8B context = 4 | unsloth/Meta-Llama-3.1-8B-Instruct | 3 | 1 | 4 | 8 | 4000 | 16 | 32 | 2e-4 |
| Llama 3.1 8B context = 4 + AG | unsloth/Meta-Llama-3.1-8B-Instruct | 4 | 1 | 4 | 8 | 6000 | 16 | 32 | 2e-4 |
| Llama 3.1 8B full + AG | unsloth/Meta-Llama-3.1-8B-Instruct | 6 | 8 | 1 | 8 | 50000 | 128 | 16 | 2e-5 |
| Llama 3.3 70B ICL | unsloth/Llama-3.3-70B-Instruct-bnb-4bit | 1 | 1 | 4 | 8 | 4000 | 16 | 32 | 2e-4 |
| Llama 3.3 70B context = 0 | unsloth/Llama-3.3-70B-Instruct-bnb-4bit | 0 | 1 | 4 | 8 | 4000 | 16 | 32 | 2e-4 |
| Llama 3.3 70B context = 2 | unsloth/Llama-3.3-70B-Instruct-bnb-4bit | 2 | 1 | 4 | 8 | 4000 | 16 | 32 | 2e-4 |
| Llama 3.3 70B context = 4 | unsloth/Llama-3.3-70B-Instruct-bnb-4bit | 3 | 1 | 4 | 8 | 4000 | 16 | 32 | 2e-4 |
| Llama 3.3 70B context = 4 + AG | unsloth/Llama-3.3-70B-Instruct-bnb-4bit | 4 | 1 | 4 | 8 | 6000 | 16 | 32 | 2e-4 |
| Llama 3.3 70B full + AG | unsloth/Llama-3.3-70B-Instruct-bnb-4bit | 6 | 1 | 1 | 4 | 50000 | 16 | 32 | 2e-5 |

Flan-T5 is trained and evaluated via `02A_train_t5.py`, e.g.:
```bash
python 02A_train_t5.py --model google/flan-t5-xxl --dataset data/windowed_dataset-replicate-LAM-0 --epochs 2 --batch_size 16 --max_seq 512
```

The Llama 3 models are trained and evaluated via `02B_train_llama.py`, e.g.:
```bash
python 02B_train_llama.py --model unsloth/Meta-Llama-3.1-8B-Instruct --output outputs/models --prediction_dir outputs/predictions --epochs 1 --batch_size 4 --gac 8 --max_seq 6000 --lora_rank 16 --lora_alpha 32 --version 4
```
`--version` selects the prompt/context scheme and the matching dataset. This runs both
training and evaluation on the test set.

## Evaluation

Before scoring, transform the raw RoBERTa (Legal-RoBERTa-15k) BIO baseline predictions from
the original LAM:ECHR release into the same in-text annotation style:
```bash
python 02C_transform_lam_baseline.py --predictions lam_predictions/original_predictions_roberta_fp_15k.csv --output outputs/paper/predictions/lam_baseline_predictions.csv
```

The prediction files to score are defined in `config.ini` (`[PREDICTIONS]`). A report can be generated with:
```bash
python 03_evaluate_predictions.py --threshold 0.1
```

## Citation
Please use the following citation:

```bibtex
@inproceedings{held-habernal-2025-contemporary,
    title     = "Contemporary {LLM}s struggle with extracting formal legal arguments",
    author    = "Held, Lena and Habernal, Ivan",
    booktitle = "Proceedings of the Natural Legal Language Processing Workshop 2025",
    year      = "2025",
    publisher = "Association for Computational Linguistics",
    pages     = "292--303",
    doi       = "10.18653/v1/2025.nllp-1.20",
}
```

## Contact
Lena Held, <lena.held@ruhr-uni-bochum.de>
