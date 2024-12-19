# Import general libraries
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import hydra 
from peft import LoraConfig, get_peft_model
from pathlib import Path
from omegaconf import OmegaConf

# Import custom libraries
from data_module import TextDatasetQA, custom_data_collator
from dataloader import CustomTrainer
from utils import get_model_identifiers_from_yaml, find_all_linear_names, print_trainable_parameters

# Import Habana libraries
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments

@hydra.main(version_base=None, config_path="config", config_name="finetune")
def main(cfg):
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
    set_seed(cfg.seed)

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    #if master process
    if os.environ.get('LOCAL_RANK') is None or local_rank == 0:
        with open(f'{cfg.save_dir}/cfg.yaml', 'w') as f:
            OmegaConf.save(cfg, f)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    num_devices = hthpu.device_count()
    ft_dataset = TextDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=500, split=cfg.split, is_local_csv=True)

    max_steps = int(cfg.num_epochs*len(ft_dataset))//(cfg.batch_size*cfg.gradient_accumulation_steps*num_devices)

    training_args = GaudiTrainingArguments(
            use_habana=True,
            use_lazy_mode=False,
            gaudi_config_name= cfg.gaudi_config_name,

            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            warmup_steps=max(1, max_steps//cfg.num_epochs),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_steps=max_steps//5,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            evaluation_strategy="no",
            deepspeed='config/ds_config.json', # deepspeed configuration usinn built-in args in Transformers
            weight_decay = cfg.weight_decay,
            seed = cfg.seed,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code = True)
    
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True

    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()

    # LoRA configuration
    if cfg.LoRA.r != 0:
        config = LoraConfig(
            r=cfg.LoRA.r, 
            lora_alpha=cfg.LoRA.alpha, 
            target_modules=find_all_linear_names(model), 
            lora_dropout=cfg.LoRA.dropout,
            bias="none", 
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        model.enable_input_require_grads()
    
    # Training using CustomTrainer from dataloader.py
    trainer = CustomTrainer(
        model=model,
        train_dataset=ft_dataset,
        eval_dataset=ft_dataset,
        args=training_args,
        data_collator=custom_data_collator,
    )
    model.config.use_cache = False  # silence the warnings.
    trainer.train()

    #save the model
    if cfg.LoRA.r != 0:
        model = model.merge_and_unload()


    model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

if __name__ == "__main__":
    main()