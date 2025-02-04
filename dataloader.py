import torch
from torch import nn
from transformers import Trainer
import torch.nn.functional as F
import copy, os
import deepspeed
from evaluate_util import get_dataloader, get_all_evals
import copy
import json 
from pathlib import Path
from data_module import get_batch_loss 
from utils import merge_dicts, interleave_eval_result_dict, get_forget_quality, get_model_utility
import numpy as np
from scipy.stats import ks_2samp, hmean
import csv 
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available

from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments

class CustomTrainer(GaudiTrainer):
    '''
    Simple custom trainer example from the fine-tuning repo
    '''
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        # forward pass
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)
    
class CustomTrainerForgetting(GaudiTrainer):
    '''
    This is the custom trainer for the MOUCHI experiment. The Trainer is derived from the GaudiTrainer (Huggingface 
    Trainer) class. The modifiction includes the following:
    1. compute_loss: We modify the compute_loss to include baslines from other experiments. Furthermore, the compute_loss also include all the losses required for the MOUCHI experiment. (forget_loss, derivative_loss, retain_loss)
    2. prediction_step: We modify the prediction_step to include the logits and labels for the MOUCHI experiment.
    3. evaluate: We modify the evaluate function to include the evaluation for the MOUCHI experiment. The evaluation includes the evaluation for the forget_rate, model utility and forget quality.

    On top of the above modifications, we also create some helper functions as follows:
    1. e_prepare_deepspeed: this function is used to prepare the oracle model for some experiments that are KL, DPO and NPO. The oracle model is used to calculate the divergence between the current model and the oracle model.
    2. log_and_print_losses: this function is used to log and print the losses for the MOUCHI experiment.
    '''
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss')
        self.oracle_model = kwargs.pop('oracle_model')
        self.eval_cfg = kwargs.pop('eval_cfg')
        self.use_drv = kwargs.pop('use_drv')
        self.derivative_loss = kwargs.pop('derivative_loss')
        self.use_rt = kwargs.pop('use_rt')
        self.retain_loss = kwargs.pop('retain_loss')

        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)
        if self.retain_loss == "KL" or self.loss_type == "DPO"  or self.loss_type == "NPO":
            self.oracle_model = self.e_prepare_deepspeed(self.oracle_model)

    def e_prepare_deepspeed(self, model):
        '''
        This code is to prepare the oracle model to be available while using the deepspeed plugin.
        The code is adapted from the accelerate library as well as the TOFU code.
        The input is the oracle model and the output is the oracle model with the deepspeed plugin.
        '''
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def log_and_print_losses(self, forget_loss, derivative_loss, retain_loss, total_loss):
        '''
        This function is used to log and print the losses according to the current experiment.
        '''
        self.state.forget_loss = forget_loss.item()
        self.state.derivative_loss = derivative_loss.item() if isinstance(derivative_loss, torch.Tensor) else derivative_loss
        self.state.retain_loss = retain_loss.item() if isinstance(retain_loss, torch.Tensor) else retain_loss
        self.state.total_loss = total_loss.item()

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                loss_msg = f"fgt_loss: {self.state.forget_loss:.3f}, tot_loss: {self.state.total_loss:.3f}"
                if self.use_drv:
                    loss_msg += f", drv_loss: {self.state.derivative_loss:.3f}"
                if self.use_rt:
                    loss_msg += f", rt_loss: {self.state.retain_loss:.3f}"
                print(loss_msg)

    def compute_loss(self, model, inputs, return_outputs=False):
        '''
        This is the modified compute_loss function for the MOUCHI experiment.
        according to the loss_type, the function will preprocess and calculate the losses accordingly.
        The inputs are the model, the state from previous steps, as well as arguments from __init__.
        '''
        if self.loss_type == "GA":
            forget_inputs, derivative_inputs, retain_inputs = inputs

             # forget loss
            input_ids, labels, attention_mask = forget_inputs
            forget_outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = forget_outputs.loss
            forget_loss = forget_loss * -1

            derivative_loss = 0
            retain_loss = 0

            # derivative loss
            if self.use_drv:
                if self.use_drv == "GD":
                    derivative_input_ids, derivative_labels, derivative_attention_mask = derivative_inputs
                    derivative_outputs = model(derivative_input_ids,labels=derivative_labels, attention_mask=derivative_attention_mask)
                    derivative_loss = derivative_outputs.loss
                elif self.use_drv == "KL":
                    derivative_input_ids, derivative_labels, derivative_attention_mask = derivative_inputs
                    with torch.no_grad():
                        derivative_outputs = self.oracle_model(derivative_input_ids,labels=derivative_labels, attention_mask=derivative_attention_mask)
                    
                    derivative_probs = F.log_softmax(derivative_outputs.logits, dim=-1)
                    derivative_probs = derivative_probs.view(-1, derivative_outputs.logits.shape[-1])

                    current_outputs = model(derivative_input_ids,labels=derivative_labels, attention_mask=derivative_attention_mask)
                    current_probs = F.log_softmax(current_outputs.logits, dim=-1)
                    current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

                    #minimum KL divergence
                    derivative_loss = nn.functional.kl_div(current_probs, derivative_probs, reduction='batchmean', log_target=True)

            # retain loss
            if self.use_rt:
                if self.retain_loss == "GD":
                    retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
                    retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                    retain_loss = retain_outputs.loss
                elif self.retain_loss == "KL":
                    retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
                    with torch.no_grad():
                        retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                    
                    retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
                    retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

                    current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                    current_probs = F.log_softmax(current_outputs.logits, dim=-1)
                    current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

                    #minimum KL divergence
                    retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)

            # total loss
            loss = forget_loss + 4 * (derivative_loss if self.use_drv else 0) +  1 * (retain_loss if self.use_rt else 0)
            self.log_and_print_losses(forget_loss, derivative_loss, retain_loss, loss)
        
        # DPO loss
        elif self.loss_type == "DPO":
            idk_inputs, forget_inputs, derivative_inputs, retain_inputs = inputs
            
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)

            with torch.no_grad():
                idk_outputs_oracle = self.oracle_model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                forget_outputs_oracle = self.oracle_model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
                idk_logits_oracle = idk_outputs_oracle.logits
                forget_logits_oracle = forget_outputs_oracle.logits

            idk_loss_oracle = -1 * get_batch_loss(idk_logits_oracle, idk_labels)
            forget_loss_oracle = -1 * get_batch_loss(forget_logits_oracle, forget_labels)
            
            idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
            forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)


            pi_logratios = idk_loss_current - forget_loss_current
            ref_logratios = idk_loss_oracle - forget_loss_oracle

            beta = 0.1
            forget_loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()

            derivative_loss = 0
            retain_loss = 0

            # derivative loss
            if self.use_drv:
                if self.derivative_loss == "GD":
                    derivative_input_ids, derivative_labels, derivative_attention_mask = derivative_inputs
                    derivative_outputs = model(derivative_input_ids,labels=derivative_labels, attention_mask=derivative_attention_mask)
                    derivative_loss = derivative_outputs.loss
                elif self.derivative_loss == "KL":
                    derivative_input_ids, derivative_labels, derivative_attention_mask = derivative_inputs
                    with torch.no_grad():
                        derivative_outputs = self.oracle_model(derivative_input_ids,labels=derivative_labels, attention_mask=derivative_attention_mask)
                    
                    derivative_probs = F.log_softmax(derivative_outputs.logits, dim=-1)
                    derivative_probs = derivative_probs.view(-1, derivative_outputs.logits.shape[-1])

                    current_outputs = model(derivative_input_ids,labels=derivative_labels, attention_mask=derivative_attention_mask)
                    current_probs = F.log_softmax(current_outputs.logits, dim=-1)
                    current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

                    #minimum KL divergence
                    derivative_loss = nn.functional.kl_div(current_probs, derivative_probs, reduction='batchmean', log_target=True)

            # retain loss
            if self.use_rt:
                if self.retain_loss == "GD":
                    retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
                    retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                    retain_loss = retain_outputs.loss
                elif self.retain_loss == "KL":
                    retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
                    with torch.no_grad():
                        retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                    
                    retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
                    retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

                    current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                    current_probs = F.log_softmax(current_outputs.logits, dim=-1)
                    current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

                    #minimum KL divergence
                    retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)

            # total loss
            loss = forget_loss + 4 * (derivative_loss if self.use_drv else 0) +  1 * (retain_loss if self.use_rt else 0)
            self.log_and_print_losses(forget_loss, derivative_loss, retain_loss, loss)            

        # NPO loss
        elif self.loss_type == "NPO":
            forget_inputs, derivative_inputs, retain_inputs = inputs

            # forget loss
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss_current = get_batch_loss(outputs.logits, labels) 

            beta = 0.1

            derivative_loss = 0
            retain_loss = 0

            with torch.no_grad():
                forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                forget_logits_oracle = forget_outputs_oracle.logits
                forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)

            neg_log_ratios = forget_loss_current - forget_loss_oracle
            forget_loss = -F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta

            # derivative loss
            if self.use_drv:
                if self.derivative_loss == "GD":
                    derivative_input_ids, derivative_labels, derivative_attention_mask = derivative_inputs
                    derivative_outputs = model(derivative_input_ids,labels=derivative_labels, attention_mask=derivative_attention_mask)
                    derivative_loss = derivative_outputs.loss
                elif self.derivative_loss == "KL":
                    derivative_input_ids, derivative_labels, derivative_attention_mask = derivative_inputs
                    with torch.no_grad():
                        derivative_outputs = self.oracle_model(derivative_input_ids,labels=derivative_labels, attention_mask=derivative_attention_mask)
                    
                    derivative_probs = F.log_softmax(derivative_outputs.logits, dim=-1)
                    derivative_probs = derivative_probs.view(-1, derivative_outputs.logits.shape[-1])

                    current_outputs = model(derivative_input_ids,labels=derivative_labels, attention_mask=derivative_attention_mask)
                    current_probs = F.log_softmax(current_outputs.logits, dim=-1)
                    current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

                    #minimum KL divergence
                    derivative_loss = nn.functional.kl_div(current_probs, derivative_probs, reduction='batchmean', log_target=True)

            # retain loss
            if self.use_rt:
                if self.retain_loss == "GD":
                    retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
                    retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                    retain_loss = retain_outputs.loss
                elif self.retain_loss == "KL":
                    retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
                    with torch.no_grad():
                        retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                    
                    retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
                    retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

                    current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                    current_probs = F.log_softmax(current_outputs.logits, dim=-1)
                    current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

                    #minimum KL divergence
                    retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)

            # total loss
            loss = forget_loss + 4 * (derivative_loss if self.use_drv else 0) +  1 * (retain_loss if self.use_rt else 0)
            self.log_and_print_losses(forget_loss, derivative_loss, retain_loss, loss)            

        elif self.loss_type == "idk":
            idk_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            
            #concatenate the inputs. single forward pass is much more efficient
            input_ids = torch.cat((idk_input_ids, retain_input_ids), dim=0)
            labels = torch.cat((idk_labels, retain_labels), dim=0)
            attention_mask = torch.cat((idk_attention_mask, retain_attention_mask), dim=0)
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
        
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(self, eval_dataset = None, ignore_keys = None, metric_key_prefix = "eval"):
        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=None)
        print(self.is_in_train, args.device, model.dtype, self.args.dataloader_num_workers, self.eval_cfg.split_list, self.eval_cfg.split)
        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        model.eval()
        curr_step = self.state.global_step
        eval_cfg = self.eval_cfg

        curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}")
        Path(curr_save_dir).mkdir(parents=True, exist_ok=True)
        forget_rate = eval_cfg.split.split('_')[0]
        with torch.no_grad():
            for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(eval_cfg.data_path, eval_cfg.split_list, eval_cfg.question_key, eval_cfg.answer_key, eval_cfg.eval_task, eval_cfg.base_answer_key, eval_cfg.perturbed_answer_key)):
                world_size = self.accelerator.num_processes

                # For some reason, Hydra is not interprating the split correctly
                if eval_task == 'eval_log_forget':
                    split = eval_cfg.split
                print(f'Working on eval task {eval_task} with split {split}')
                save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                save_filename = save_filename if world_size == 1 else os.path.join(curr_save_dir, f"{eval_task}_{self.accelerator.local_process_index}.json")
                # print(save_filename)
                if os.path.exists(save_filename) and not eval_cfg.overwrite:
                    print(f"Skipping {eval_task} because {save_filename} already exists")
                    continue

                eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(eval_cfg, eval_task, self.tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)
                eval_dataloader = self.accelerator.prepare(eval_dataloader)
                # print('dataset condition: ', len(eval_dataloader.dataset), self.accelerator.local_process_index)
                base_eval_dataloader = self.accelerator.prepare(base_eval_dataloader)
                perturb_dataloader = self.accelerator.prepare(perturb_dataloader)
                normalize_gt = False 
                # if 'eval_log' not in eval_task:
                #     normalize_gt = True

                eval_logs = get_all_evals(eval_cfg, model, self.tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=normalize_gt)

                with open(save_filename, "w") as f:
                    # pretty write json to f
                    json.dump(eval_logs, f, indent=4)
            
                #wait for all process to finish
            self.accelerator.wait_for_everyone()
            aggregated_eval_logs = {}
            for eval_task in eval_cfg.eval_task:
                #read the saved file as json and merge them using merge_dicts
                if world_size > 1:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}_0.json")))
                        for i in range(1, world_size):
                            filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                            eval_logs = merge_dicts(eval_logs, json.load(open(filename)))
                        
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

                        new_save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                        with open(new_save_filename, "w") as f:
                            # pretty write json to f
                            json.dump(eval_logs, f, indent=4)

                            #delete old files use shutil

                            for i in range(world_size):
                                filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                                os.remove(filename)
                                
            if self.accelerator.is_local_main_process:
                # aggregated_eval_logs = interleave_eval_result_dict(aggregated_eval_logs, forget_rate, large_bsz=eval_cfg.batch_size, num_processes=world_size)
                aggregated_eval_log_filename = os.path.join(curr_save_dir, "eval_log_aggregated.json")

                with open(aggregated_eval_log_filename, 'w') as f:
                    json.dump(aggregated_eval_logs, f, indent=4)

                if eval_cfg.retain_result is not None:
                    model_utility = get_model_utility(aggregated_eval_logs)
                    retain_result = json.load(open(eval_cfg.retain_result, 'r'))
                    forget_quality = get_forget_quality(aggregated_eval_logs, retain_result)
                    aggregate_stat = {**model_utility, **forget_quality}

                    # save aggregate_stat as csv
                    with open(os.path.join(curr_save_dir, "aggregate_stat.csv"), 'w') as csvfile:
                        field_names = list(aggregate_stat.keys())
                        writer = csv.DictWriter(csvfile, fieldnames=field_names)
                        writer.writeheader()
                        writer.writerow(aggregate_stat)

def custom_data_collator_forget(samples):
    forget_samples, derivative_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples], [sample[2] for sample in samples]
    rets = []
    for data_type in ["forget", "derivative", "retain"]:
        if data_type == "forget":
            data = forget_samples
        elif data_type == "derivative":
            data = derivative_samples
        else:
            data = retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets

def custom_data_collator_dpo(samples):
    idk_samples, forget_samples, derivative_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples], [sample[2] for sample in samples], [sample[3] for sample in samples]
    data_types = ["idk", "forget", "derivative", "retain"]

    rets = []
    for data_type in data_types:
        if data_type == "idk":
            data = idk_samples
        elif data_type == "forget":
            data = forget_samples
        elif data_type == "derivative":
            data = derivative_samples
        else:
            data = retain_samples

        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))

    return rets