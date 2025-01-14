# Import general libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import hydra 
import os
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path

# Import custom libraries
from data_module import TextForgetDatasetQA, TextForgetDatasetDPOQA, TextDatasetQA, TextForgetDrvDatasetQA
from dataloader import CustomTrainerForgetting, custom_data_collator_forget, custom_data_collator_dpo
from utils import setup_model, find_all_linear_names, print_trainable_parameters

# Import Habana libraries
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
from optimum.habana import GaudiTrainingArguments

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    '''
    the main function for doing forgetting with or without MOUCHI. 
    The code is relatively similar to the finetune.py code, but with using the CustomTrainerForgetting instead of the CustomTrainer class.
    Therefore, check the CustomTrainerForgetting class for more details on the implementation.
    '''

    # setup the model
    model_cfg, model_id = setup_model(cfg)

    # Wandb setup
    os.environ["WANDB_DISABLED"] = "true"

    # set the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # set the datasets and data collator
    if cfg.forget_loss == "DPO":
        train_dataset = TextForgetDatasetDPOQA(cfg.forget_data_path, cfg.derivative_data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=500, fgt_local_csv=True, drv_local_csv=True)
        data_collator = custom_data_collator_dpo
    else:
        train_dataset = TextForgetDrvDatasetQA(cfg.forget_data_path, cfg.derivative_data_path, cfg.retain_data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=500, fgt_local_csv=True, drv_local_csv=True)
        data_collator = custom_data_collator_forget

    
    num_devices = hthpu.device_count()
    steps_per_epoch = len(train_dataset)//(cfg.batch_size*cfg.gradient_accumulation_steps*num_devices)
    max_steps = int(cfg.num_epochs*len(train_dataset))//(cfg.batch_size*cfg.gradient_accumulation_steps*num_devices)
    print(f"steps_per_epoch: {steps_per_epoch}")
    print(f"max_steps: {max_steps}")

    # set the training arguments, including the Gaudi specific arguments
    training_args = GaudiTrainingArguments(
            use_habana=True,
            use_lazy_mode=False,
            gaudi_config_name= "Habana/llama",
            
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            warmup_steps=0,
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//5),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_strategy="steps" if cfg.save_model and (not cfg.eval_only) else "no",
            save_steps=steps_per_epoch//4,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            deepspeed='config/ds_config.json',
            weight_decay = cfg.weight_decay,
            eval_steps = steps_per_epoch,
            evaluation_strategy = "steps" if cfg.eval_while_train else "no",
            seed=cfg.seed

        )
    
    #first get the base model architecture
    #if there is a pytorch*.bin file in the model path, then load that. use regex there can be anythign in between pytorch and .bin
    import re
    path_found = False
    for file in os.listdir(cfg.model_path):
        if re.search("pytorch.*\.bin", file):
            path_found = True
            break
        
        if re.search("model-*\.safetensors", file):
            path_found = True
            break

    oracle_model = None

    if path_found:
        config = AutoConfig.from_pretrained(model_id)
        print("Loading from checkpoint")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code = True)
        if cfg.retain_loss == "KL" or cfg.forget_loss == "DPO" or cfg.forget_loss == "NPO":
            oracle_model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code = True)

    else:
        print("Loading after merge and unload")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device_map)
        #now use the checkpoint to add the LoRA modules
        model = PeftModel.from_pretrained(model, model_id = cfg.model_path)
        #save this as a standard model so that we can again do PEFT style finetuneing from scratch
        model = model.merge_and_unload()
        #save the model for next time
        model.save_pretrained(cfg.model_path)
    
    
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True
    
    #now we have a HuggingFace model 
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
    config = LoraConfig(
        r=cfg.LoRA.r, 
        lora_alpha=cfg.LoRA.alpha, 
        target_modules=find_all_linear_names(model), 
        lora_dropout=cfg.LoRA.dropout,
        bias="none", 
        task_type="CAUSAL_LM"
    )
    if cfg.LoRA.r != 0:
        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    # Use CustomTrainerForgetting for the forgetting task
    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset = train_dataset,
        args=training_args,
        data_collator=data_collator,
        oracle_model = oracle_model,
        forget_loss = cfg.forget_loss,
        retain_loss = cfg.retain_loss,
        derivative_loss = cfg.derivative_loss,
        use_drv = cfg.use_drv,
        use_rt = cfg.use_rt,
        eval_cfg = cfg.eval,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    # trainer.train()
    if cfg.eval_only:
        trainer.evaluate()
    else:
        print("Training")
        trainer.train()

    #save the tokenizer
    if cfg.save_model and (not cfg.eval_only):
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)

    #delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                #delete the directory
                import shutil
                shutil.rmtree(global_step_dir)



if __name__ == "__main__":
    main()