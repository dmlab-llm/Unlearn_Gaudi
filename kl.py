import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.special import kl_div
from transformers import AutoTokenizer
import torch

def calculate_kl_divergence(forget_dataset, delta_min_path, delta_max_path, model_name):
    # Load datasets
    main_dataset = pd.read_csv(forget_dataset, delimiter=';')
    delta_min = pd.read_csv(delta_min_path, delimiter=';')
    delta_max = pd.read_csv(delta_max_path, delimiter=';')

    # Concatenate question and answer columns
    main_dataset['text'] = main_dataset['question'] + ' ' + main_dataset['answer']
    delta_min['text'] = delta_min['question'] + ' ' + delta_min['answer']
    delta_max['text'] = delta_max['question'] + ' ' + delta_max['answer']

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Tokenize the datasets
    X = tokenizer(main_dataset['text'].tolist(), truncation=True, padding=True, return_tensors='pt')
    Y_min = tokenizer(delta_min['text'].tolist(), truncation=True, padding=True, return_tensors='pt')
    Y_max = tokenizer(delta_max['text'].tolist(), truncation=True, padding=True, return_tensors='pt')

    # Get the entire vocabulary
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)

    # Calculate frequency distributions
    main_dataset_freq = torch.bincount(X['input_ids'].flatten(), minlength=vocab_size) + 1
    main_dataset_freq = main_dataset_freq / main_dataset_freq.sum()
    delta_min_freq = torch.bincount(Y_min['input_ids'].flatten(), minlength=vocab_size) + 1
    delta_min_freq = delta_min_freq / delta_min_freq.sum()
    delta_max_freq = torch.bincount(Y_max['input_ids'].flatten(), minlength=vocab_size) + 1
    delta_max_freq = delta_max_freq / delta_max_freq.sum()

    # Calculate KL divergences
    kl_divergence_min = kl_div(main_dataset_freq, delta_min_freq).sum()
    kl_divergence_max = kl_div(main_dataset_freq, delta_max_freq).sum()

    return kl_divergence_min, kl_divergence_max

if __name__ == "__main__":
    forget_dataset = '../path/to/forget'
    delta_min_path = '../path/to/delta_min'
    delta_max_path = '../path/to/delta_max'
    model_name = "../path/to/model"

    kl_divergence_min, kl_divergence_max = calculate_kl_divergence(forget_dataset, delta_min_path, delta_max_path, model_name)

    print(f'KL divergence for forget to retain: {kl_divergence_min}')
    # print(f'KL divergence for delta_max: {kl_divergence_max}')