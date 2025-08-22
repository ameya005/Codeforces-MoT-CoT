# Codeforces eval and MoT training

This repository contains a comprehensive framework for training and fine-tuning large language models (LLMs) for competitive programming tasks, with a focus on the Open-R1 evaluation framework.

## Overview

The project implements various training approaches including:
- Supervised Fine-Tuning (SFT) with LoRA adapters
- Knowledge Distillation (KL) training
- Full-parameter fine-tuning with DeepSpeed
- Data preprocessing for competitive programming datasets

## Project Structure

```
├── scripts/           # Training and utility scripts
├── src/              # Source code for evaluation
├── data/             # Dataset storage
├── outputs/          # Evaluation outputs
├── checkpoints/      # Model checkpoints
├── storage/          # Additional storage for large files
├── wandb/            # Weights & Biases logs
└── configs/          # Configuration files
```

## Scripts Documentation

### Core Training Scripts

#### `scripts/sft_train.py`
**Purpose**: Main script for supervised fine-tuning of language models.

**Key Features**:
- Supports both LoRA and full-parameter fine-tuning
- DeepSpeed integration for distributed training
- Automatic checkpoint resumption
- Weights & Biases logging with custom callbacks
- Token-per-second monitoring

**Usage**:
```bash
# LoRA training
accelerate launch --num_processes 8 scripts/sft_train.py \
  --dataset_name /path/to/dataset.jsonl \
  --custom_dataset \
  --model_id Qwen/Qwen2-7B-Instruct \
  --use_lora --lora_r 16 --lora_alpha 32 \
  --max_seq_len 8192 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 --num_train_epochs 3 \
  --deepspeed scripts/configs/ds_z2.json \
  --output_dir /path/to/output
```

#### `scripts/sft_train_kl.py`
**Purpose**: SFT training with Knowledge Distillation (KL divergence) loss.

**Key Features**:
- Combines standard SFT loss with KL divergence from a reference model
- Configurable KL coefficient and temperature
- Automatic vocabulary alignment between reference and student models
- Supports 8-bit quantization for reference models

**Usage**:
```bash
accelerate launch --num_processes 4 scripts/sft_train_kl.py \
  --dataset_name /path/to/dataset.jsonl \
  --model_id Qwen/Qwen2-7B-Instruct \
  --kl_coef 0.1 --kl_t 1.0 \
  --kl_ref_model_id /path/to/reference/model \
  --output_dir /path/to/output
```

#### `scripts/klsft.py`
**Purpose**: Custom trainer class implementing KL divergence loss for SFT.

**Key Features**:
- Extends `SFTTrainer` with KL divergence functionality
- Handles vocabulary mismatches between models
- Efficient reference model management
- Custom loss computation with configurable parameters

### Data Processing Scripts

#### `scripts/preprocess_mot.py`
**Purpose**: Preprocesses competitive programming datasets for training.

**Key Features**:
- Extracts chain-of-thought reasoning from model outputs
- Generates training examples with system prompts
- Supports multiple programming languages (Python, C++, etc.)
- Batch processing with progress tracking
- Automatic dataset sharding for distributed processing

**Usage**:
```bash
python scripts/preprocess_mot.py \
  --input_file /path/to/input.jsonl \
  --output_file /path/to/output.jsonl \
  --model_id /path/to/model \
  --max_examples 1000
```

### Utility Scripts

#### `scripts/merge_lora.py`
**Purpose**: Merges LoRA adapters with base models for inference.

**Key Features**:
- Combines LoRA weights with base model
- Handles vocabulary size mismatches
- Supports different precision formats (bfloat16, float16)
- Automatic tokenizer alignment

**Usage**:
```bash
python scripts/merge_lora.py \
  --base_model Qwen/Qwen2-7B-Instruct \
  --adapter_path /path/to/lora/checkpoint \
  --output_dir /path/to/merged/model \
  --dtype bfloat16
```


## Makefile Commands

The project includes a comprehensive Makefile for common operations:

### Training Commands
- `make sft-lora`: Train with LoRA adapters
- `make sft-full-ds`: Full-parameter training with DeepSpeed
- `make kl-sft-lora`: KL training with LoRA

### Evaluation Commands
- `make eval-log`: Run evaluation with different pass counts
- `make eval-checkpoint`: Evaluate a specific checkpoint
- `make eval-converted`: Evaluate a converted checkpoint

### Utility Commands
- `make convert-checkpoint`: Convert DeepSpeed checkpoint
- `make convert-checkpoint-custom`: Convert with custom paths

## Configuration

### DeepSpeed Configurations
- `scripts/configs/ds_z2.json`: ZeRO stage 2 configuration
- `scripts/configs/ds_z3.json`: ZeRO stage 3 configuration

### Environment Setup
The project uses Conda for environment management:
```bash
conda env create -f conda_env.yaml
conda activate openr1
pip install -r requirements-openr1.txt
```

## Key Dependencies

- **PyTorch**: 2.6.0
- **Transformers**: >=4.49.0
- **TRL**: 0.18.0 (with VLLM support)
- **DeepSpeed**: 0.15.4
- **PEFT**: >=0.14.0
- **Accelerate**: 1.4.0

## Training Workflow

1. **Data Preparation**: Use `preprocess_mot.py` to prepare Mixture-of-thought dataset
2. **Model Training**: Choose between LoRA/full-ds (`sft_train.py`) or KL training (`sft_train_kl.py`)
3. **Checkpoint Management**: Use `utils.py` to convert DeepSpeed checkpoints if needed
4. **Model Merging**: Use `merge_lora.py` to combine LoRA adapters with base models
5. **Evaluation**: Use the evaluation framework in `src/open_r1/evals/`

