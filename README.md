# Mitigating Over-forgetting in Unlearning Copyrighted Information (MOUCHI) in Gaudi-v2

## Introduction

In this repository, we implement the MOUCHI framework in Gaudi-v2. The framework consists of two main submodules:

1. **Derivative Knowledge Generation**
2. **Derivative Knowledge Incorporation During the Unlearning Process**

This code is part of our research paper and is built upon the original code provided by the TOFU dataset used in our experiments [[1]](https://github.com/dmlab-llm/Unlearn_Gaudi/tree/main#1). For additional details, we encourage you to explore their repository.

## Getting Started

Inside the Gaudi-v2 docker environment, install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

## Fine-tuning

To fine-tune the model on the dataset, run the following command:

```bash
PT_HPU_LAZY_MODE=0 python finetune.py --config-name=finetune_lora.yaml
```

**Explanation:**
- `PT_HPU_LAZY_MODE=0`: Enables eager mode as lazy mode is currently not supported on Gaudi-v2.
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

## Detailed Explanation

### forget.py

The main code for forgetting. However, the content is similar to a custom fine-tuning process. The only difference is the usage of the `CustomTrainerForgetting` class from `dataloader.py`:

```python
trainer = CustomTrainerForgetting(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    args=training_args,
    data_collator=data_collator,
    oracle_model=oracle_model,
    forget_loss=cfg.forget_loss,
    retain_loss=cfg.retain_loss,
    derivative_loss=cfg.derivative_loss,
    use_drv=cfg.use_drv,
    use_rt=cfg.use_rt,
    eval_cfg=cfg.eval,
)
```

### dataloader.py

**`class CustomTrainer(Gaudi-v2Trainer)`**

The custom trainer used for fine-tuning the model with the dataset throughout the experiment.

**`class CustomTrainerForgetting(Gaudi-v2Trainer)`**

Consists of several functions:

- `__init__(self, *args, **kwargs)`: Parameters passed from the `Gaudi-v2TrainingArgs`.
- `e_prepare_deepspeed(self, model)`: Modified code from the original transformer’s DeepSpeed code to prepare DeepSpeed.
- `log_and_print_losses(self, forget_loss, derivative_loss, retain_loss, total_loss)`: Loss-printing function.
- `compute_loss(self, model, inputs, return_outputs=False)`: Modified compute_loss function from the original Hugging Face code to accommodate derivative loss.
- `custom_data_collator_forget(samples)` and `custom_data_collator_dpo(samples)`: Custom data collators due to multiple types of input (forget, retain, derivative).

### data_module.py

- `class TextForgetDatasetQA(Dataset)` and `class TextForgetDrvDatasetQA(Dataset)`:
  Convert the dataset into a suitable format for Llama 2.

- `convert_raw_data_to_model_format(tokenizer, max_length, question, answer, model_configs)`:
  Convert dataset text to the model (Llama 2) format using their tokenizer and special token marks.

- `custom_data_collator`:
  Custom collator to combine the three subsets (forget, derivative, retain) during unlearning.

### utils.py

- `get_model_identifiers_from_yaml`: Helper function for getting the model-specific token tag for converting text into tokens.
- `find_all_linear_names`: Helper function to get all linear layers for LoRA.
- `print_trainable_parameters`: Printing function for LoRA.
- `get_model_utility(eval_result_dict)` and `get_forget_quality(unlearn_result, retain_result)`: Helper functions for evaluating unlearning performance.
- `setup_model(cfg)`: Helper function for model preprocessing for both fine-tuning and unlearning.

### Config Files

All parameters can be configured in the config files:

#### [finetune_lora.yaml](https://github.com/dmlab-llm/Unlearn_Gaudi-v2/blob/main/config/finetune_lora.yaml)

- `model_id`: HF model used for fine-tuning (default: `Llama-2-7b-chat-hf`)
- `model_family`: Llama 2 7B
- `LoRA`: LoRA parameter settings
- `data_path`: Path to the dataset for fine-tuning
- `batch_size`: Fine-tuning batch size
- `gradient_accumulation_steps`: Gradient accumulation steps
- `num_epochs`: Number of epochs
- `save_dir`: Save directory
- `lr`: Learning rate
- `weight_decay`: Weight decay
- `seed`: Seed for reproducibility

#### [forget.yaml](https://github.com/dmlab-llm/Unlearn_Gaudi-v2/blob/main/config/forget.yaml)

- `forget_data_path`: Path to forget dataset
- `derivative_data_path`: Path to derivative dataset
- `retain_data_path`: Path to retain dataset
- `forget_loss`: Forget loss used
- `retain_loss`: Retain loss used
- `derivative_loss`: Derivative loss used
- `use_drv`: Whether unlearning includes derivative loss
- `use_rt`: Whether unlearning includes retain loss


## Reference

[1] Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C. Lipton, and J. Zico Kolter. TOFU: A task of fictitious unlearning for LLMs. In Proceedings of the Conference on Language Modeling (COLM), 2024.
