import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
from utils import get_model_identifiers_from_yaml, add_dataset_index

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    '''
    This function converts the raw data to the format required by the model depending on the model.
    For instance, Llama-2-7b-chat-hf requires the question to be wrapped with the question start of [INST] and end tokens [/INST].
    This function will first wrap the question with the start and end tokens and then tokenize the question and answer.
    '''
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)
    
class TextForgetDatasetQA(Dataset):
    '''
    This class is used to preprocess and load the dataset for the forgetting task
    The class inherits from the torch Dataset class.
    The class can takes either a local csv file or a dataset from the Huggingface datasets library.

    For each QnA Pair, this class will:
    1. Load the dataset accordinf to the subset (forget, retain, idk)
    2. Give the index according to the subset
    3. Convert the raw data to the format required by the model using the convert_raw_data_to_model_format function
    4. Pad the input_ids, labels and attention_mask to the max_length
    5. Return the list of padded input_ids, labels, attention_mask and the index of the dataset as a tensor
    '''
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget10", is_local_csv=False, loss_type="idk" ,include_retain=True):
        super(TextForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        if is_local_csv:
            self.forget_data = datasets.load_dataset('csv', data_files=data_path, delimiter=';')["train"]
        else:
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        self.split1 = "idk" if self.loss_type == "idk" else "forget"
        self.split2 = "retain" if include_retain else self.split1

        if include_retain:
            retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]

        if self.loss_type == "idk":
            self.idontknowfile = "data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            question = data[idx]['question']
            answer = data[idx]['answer']

            if data_type == "idk":
                #get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets

class TextForgetDrvDatasetQA(Dataset):
    '''
    This class have the same functionality as TextForgetDatasetQA but it also includes the derivative data. 
    Therefore, the behavior is the same as TextForgetDatasetQA except that it includes the derivative data.
    '''
    def __init__(self, fgt_data_path=None, drv_data_path=None, retain_data_path=None, tokenizer=None, model_family=None,  max_length=512, fgt_local_csv=False, drv_local_csv=False):
        super(TextForgetDrvDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = datasets.load_dataset('csv', data_files=fgt_data_path, delimiter=';')["train"] if fgt_local_csv else datasets.load_dataset(fgt_data_path)["train"]
        self.derivative_data = datasets.load_dataset('csv', data_files=drv_data_path, delimiter=';')["train"] if drv_local_csv else datasets.load_dataset(drv_data_path)["train"]
        self.retain_data = datasets.load_dataset('csv', data_files=drv_data_path, delimiter=';')["train"]
        self.model_configs = get_model_identifiers_from_yaml(model_family)

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        # get the data from the forget, derivative and retain datasets
        for data_type in ["forget", "derivative", "retain"]:
            if data_type == "derivative":
                data = self.derivative_data
            elif data_type == "forget":
                data = self.forget_data
            elif data_type == "retain":
                data = self.retain_data

            # get the index according to the dataset
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            question = data[idx]['question']
            answer = data[idx]['answer']
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets

class TextForgetDatasetDPOQA(Dataset):
    '''
    This class also have the same functionality as TextForgetDatasetQA.
    The diffence is the inclusion of idontknowfile that contains idk-variant answers such as "I don't know", "I have no idea", etc.
    This class is specifically used for the DPO task.
    '''
    def __init__(self, fgt_data_path=None, drv_data_path=None, retain_data_path=None, tokenizer = None, model_family = None, max_length=512,  fgt_local_csv=False, drv_local_csv=False):
        super(TextForgetDatasetDPOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = datasets.load_dataset('csv', data_files=fgt_data_path, delimiter=';')["train"] if fgt_local_csv else datasets.load_dataset(fgt_data_path)["train"]
        self.derivative_data = datasets.load_dataset('csv', data_files=drv_data_path, delimiter=';')["train"] if drv_local_csv else datasets.load_dataset(drv_data_path)["train"]
        self.retain_data = datasets.load_dataset('csv', data_files=drv_data_path, delimiter=';')["train"]
        self.idontknowfile = "./data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "derivative", "retain"]:
            if data_type == "derivative":
                data = self.derivative_data
            elif data_type == "forget" or data_type == "idk":
                data = self.forget_data
            elif data_type == "retain":
                data = self.retain_data

            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            
            question = data[idx]['question']
            
            if data_type != "idk":
                answer = data[idx]['answer']
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets

class TextDatasetQA(Dataset):
    '''
    This class is used to preprocess and load the dataset for the QA task for the finetuning process. 
    The class inherits from the torch Dataset class.
    The class can takes either a local csv file or a dataset from the Huggingface datasets library.

    For each QnA Pair, this class will:
    1. Load the dataset
    2. Convert the raw data to the format required by the model using the convert_raw_data_to_model_format function
    3. Pad the input_ids, labels and attention_mask to the max_length
    4. Return the padded input_ids, labels, attention_mask and the index of the dataset as a tensor
    '''
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, is_local_csv=False, question_key='question', answer_key='answer'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        if is_local_csv:
            self.data = datasets.load_dataset('csv', data_files=data_path, delimiter=';')["train"]
        else:
            self.data = datasets.load_dataset(data_path, split)["train"]
        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]
        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)

def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks

def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

def custom_data_collator_with_indices(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss