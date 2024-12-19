# MOUCHI: Mitigating Over-forgetting in Unlearning Copyrighted Information

This code is part of our paper and is built upon the original code provided by the TOFU dataset used in our experiments [[1]](#1). For additional details, we encourage you to explore their repository as well

## Installation

```
conda create -n tofu python=3.10
conda activate tofu
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Finetune your models

This script is used to fine-tune the model in preparation for the unlearning process. for fine-tuning using lora, use ./config/finetune-lora.yaml

```
master_port=18765
lr=1e-4
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port finetune.py --config-name=finetune.yaml  batch_size=4 gradient_accumulation_steps=4 model_family=llama2-7b lr=${lr}
```

## Forget models

This script is for forgetting the finetuned mode:

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=llama2-7b lr=${lr}
```

## Generation and KL
all the generation code using llama and gpt with its prompt are available in the generation folder. Furthermore, the KL.py is utilized for calculating the KL values throughout the experiment

## Evaluate models
all the code used for evaluation is available in the eval folder. Example for the evaluate_util script
```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$port evaluate_util.py\
 model_family=llama2-7bsplit=$split\
 model_path=$model_path
```
You can modify the configuration in config/eval_everything.yaml

The evaluation result will by default be dumped to `${model_path}/eval_results/ds_size${ds_size}`, you can also modify the `save_dir` field in `config/eval_everything.yaml`

The evaluation results on three datasets (forget, retain, normal) will be aggregated into one json file named `eval_log_aggregated.json`


## Datasets

all datasets that we use throughout experiment are available in the data folder



## Reference

<a id="1">[1]</a> 
Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C. Lipton, and J. Zico Kolter.
TOFU: A task of fictitious unlearning for LLMs.
In Proceedings of the Conference on Language Modeling (COLM), 2024.
