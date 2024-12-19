# Mitigating Over-forgetting in Unlearning Copyrighted Information (MOUCHI) in Gaudi

## Introduction

In this repository, we implement the MOUCHI framework in Gaudi. The framework consists of two main submodules:

1. **Derivative Knowledge Generation**
2. **Derivative Knowledge Incorporation During the Unlearning Process**

This code is part of our research paper and is built upon the original code provided by the TOFU dataset used in our experiments [[1]](https://github.com/dmlab-llm/Unlearn_Gaudi/tree/main#1). For additional details, we encourage you to explore their repository.

## Getting Started

Inside the Gaudi docker environment, install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

## Fine-tuning

To fine-tune the model on the dataset, run the following command:

```bash
PT_HPU_LAZY_MODE=0 python finetune.py --config-name=finetune_lora.yaml
```

**Explanation:**
- `PT_HPU_LAZY_MODE=0`: Enables eager mode as lazy mode is currently not supported on Gaudi.
- `--config-name`: Specifies the configuration file. Detailed configurations can be found in `config/finetune_lora.yaml`.

For fine-tuning using LoRA, use the configuration file located at `./config/finetune_lora.yaml`.

## Forgetting Models

To perform the unlearning process on the fine-tuned model, use the following script:

```bash
PT_HPU_LAZY_MODE=0 python forget_drv.py --config-name=forget.yaml
```

The parameters, including the unlearning hyperparameters and loss functions, can be configured in `config/forget.yaml`.

## Derivative Generation and KL Divergence

To generate derivative knowledge, run the following script:

```bash
PT_HPU_LAZY_MODE=0 python generate_drv.py --args
```

**Arguments:**
- `--ft_path`: Path to the fine-tuned model (default: `./path/to/finetuned/model`)
- `--input_csv`: Path to the input CSV file containing the data (default: `./path/to/input/csv`)
- `--output_csv`: Path to save the output CSV file (default: `./path/to/output/csv`)
- `--delta_min`: Minimum delta value for derivative generation (default: `0.1`)
- `--delta_max`: Maximum delta value for derivative generation (default: `0.5`)
- `--shard_size`: Number of samples per shard (default: `20`)

Additionally, the `KL.py` script is used to calculate KL divergence values throughout the experiment.

## Reference

[1] Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C. Lipton, and J. Zico Kolter. TOFU: A task of fictitious unlearning for LLMs. In Proceedings of the Conference on Language Modeling (COLM), 2024.
