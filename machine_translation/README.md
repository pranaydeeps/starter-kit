# Machine Translation Starter Kit

This directory contains notebooks and utilities for training and evaluating Neural Machine Translation (NMT) models.

## 📚 Contents

| File                               | Description                                               |
| ---------------------------------- | --------------------------------------------------------- |
| `llm-finetune-translation.ipynb` | Fine-tune an LLM (Mistral-7B) for translation using QLoRA |
| `nmt-opennmt-training.ipynb`     | Train an encoder-decoder Transformer using OpenNMT-py     |
| `mt-evaluation.ipynb`            | Evaluate translations with BLEU, chrF++, TER, and COMET   |
| `download_data.py`               | Download parallel data from Hugging Face Hub              |
| `tok_detok.py`                   | Tokenization/detokenization utility (XLM-RoBERTa)         |

## 🚀 Quick Start

### 1. Choose Your Approach

| Approach                          | Best For                                             | GPU Memory    |
| --------------------------------- | ---------------------------------------------------- | ------------- |
| **LLM Fine-tuning**         | Leveraging pre-trained knowledge, few-shot scenarios | ~16GB (4-bit) |
| **OpenNMT Encoder-Decoder** | Full control, research, production NMT               | ~4-8GB        |

### 2. Install Dependencies (preferably in separate venvs)

```bash
# For LLM fine-tuning
pip install -r requirements-llm.txt

# For OpenNMT training
pip install -r requirements-opennmt.txt

# For evaluation only
pip install -r requirements-evaluation.txt
```

### 3. Download Data

```bash
python download_data.py \
    --repo_name LT3/nfr_bt_nmt_english-french \
    --base_path data/en-fr
```

## 📓 Notebooks Overview

### LLM Fine-tuning (`llm-finetune-translation.ipynb`)

Fine-tune `mistralai/Mistral-7B-Instruct-v0.3` for English→French translation:

- **Method**: QLoRA (4-bit quantization + LoRA adapters)
- **Framework**: Hugging Face Transformers + TRL
- **Data format**: Chat template with instruction prompt
- **Training**: SFTTrainer with efficient batching

### OpenNMT Training (`nmt-opennmt-training.ipynb`)

Train a Transformer encoder-decoder model from scratch:

- **Framework**: OpenNMT-py
- **Tokenization**: XLM-RoBERTa subword tokenization
- **Architecture**: 4-layer Transformer (configurable)
- **Features**: YAML configs, vocabulary building, inference

### Evaluation (`mt-evaluation.ipynb`)

Evaluate MT outputs with standard metrics:

| Metric | Type              | Higher/Lower |
| ------ | ----------------- | ------------ |
| BLEU   | N-gram precision  | Higher ↑    |
| chrF++ | Character n-grams | Higher ↑    |
| TER    | Edit distance     | Lower ↓     |
| COMET  | Neural metric     | Higher ↑    |

## 🛠️ Utilities

### `download_data.py`

Download parallel corpora from Hugging Face:

```python
from download_data import download_and_save_dataset
paths = download_and_save_dataset("LT3/nfr_bt_nmt_english-french", "data/en-fr")
```

### `tok_detok.py`

Tokenize/detokenize text files:

```python
from tok_detok import TokenDetokenizer

tok = TokenDetokenizer(model_name="xlm-roberta-base")
tok.tokenize("input.txt", "input.tok")
tok.detokenize("output.tok", "output.txt")
```

Command line:

```bash
python tok_detok.py --input data/ --mode tokenize
python tok_detok.py --input output.tok --mode detokenize
```

## 📊 Evaluation Quick Reference

```python
import sacrebleu

# BLEU
bleu = sacrebleu.corpus_bleu(hypotheses, [references])

# chrF++
chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)

# TER
ter = sacrebleu.corpus_ter(hypotheses, [references])
```

Command line:

```bash
sacrebleu reference.txt -i hypothesis.txt -m bleu chrf ter
```

## ⚠️ Notes

- **Small datasets**: Reduce `bucket_size` in OpenNMT configs (see notebook)
- **GPU memory**: LLM fine-tuning requires 4-bit quantization for 7B models
- **COMET**: Requires ~1.5GB model download, benefits from GPU
