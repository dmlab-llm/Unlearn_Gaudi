import csv
import argparse
import torch
import re
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu


def remove_question_from_answer(question, generated_text):
    """Function to post-process generated text to remove the question"""
    return generated_text[len(question):].strip() if generated_text.startswith(question) else generated_text

def clean_generated_answer(generated_answer):
    # Use regex to extract the question and answer
    question_match = re.search(r'Question: (.*?)\n', generated_answer, re.DOTALL)
    answer_match = re.search(r'Answer: (.*?)\n', generated_answer, re.DOTALL)
    
    question = question_match.group(1).strip() if question_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    
    # Remove any reference part in parentheses at the end of the answer
    answer = re.sub(r'\s*\(Reference:.*\)$', '', answer)
    
    return f"{question};{answer}"

def generate_answer(question, answer, delta_max, delta_min, generator, shard):
    prompt =  f"""
    <<SYS>>
    You are generating a question and answer for a dataset named 'Derivative Knowledge Dataset.' This questions and answers is derived from an existing dataset originally designed for unlearning copyrighted content related to book authors. The original dataset includes question and answer pairs about the authors and their book content. To effectively create the new dataset while adhering to fair use, include snippets from the original dataset as references.

    subset of dataset:
    {shard}
    
    Instructions:
    1. Inclusion of Original Data Snippets: Provide a snippet from the original dataset as a reference for each new question and answer pair you create. Ensure that these snippets are used to derive broader, non-specific questions that fall within legal bounds
    2. Delta Bounds: Create questions based on two categories - delta_max (upper boundary of knowledge that can be retained without infringement) and delta_min (lower boundary of essential knowledge). Assume a KL divergence of X for delta_max and Y for delta_min between this new set and the original dataset, where X and Y are your specific KL divergence values.
    3. KL Divergence Use: The KL divergence values provided ({delta_max} for delta_max and {delta_min} for delta_min) guide the specificity and depth of your questions, ensuring they fall within the legal bounds of derivative knowledge.

    generate the question and answer in the following format:
    question;answer
    <</SYS>>

    [INST]
    User: Generate the derivative knowldge question and answer pair for the given question and answer.
    Question: {question}
    Answer: {answer}

    [/INST]\n

    Assistant:

     """

    generated_answer = generator(prompt, max_length=2048, do_sample=True, return_full_text=False)[0]['generated_text']
    clean_answer = remove_question_from_answer(question, generated_answer)

    return clean_answer
    

def main(args):
    """Main function to load model, generate answers and write to CSV"""

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("hpu")
    print(f"Device: {device}")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.ft_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")

    # Load the model and generator
    model = AutoModelForCausalLM.from_pretrained(args.ft_path,  torch_dtype=torch.bfloat16)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)

    # Read the CSV file and generate answers
    questions_answers = []
    delta_max = args.delta_max
    delta_min = args.delta_min
    shard_size = args.shard_size # Number of questions per authors
    with open(args.input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader)  # Skip the header row
        dataset = list(reader)

    for i, row in enumerate(dataset):
        question, answer = row

        # Determine the current shard based on the current index and shard_size
        shard_start = (i // shard_size) * shard_size  
        shard_end = min(shard_start + shard_size, len(dataset))  
        shard = dataset[shard_start:shard_end]  

        generated_answer = generate_answer(question, answer, delta_max, delta_min, generator, shard)
        generated_answer = clean_generated_answer(generated_answer)
        gen_question, gen_answer = generated_answer.split(';')
        questions_answers.append((gen_question, gen_answer))

    # Write the new CSV file with question and answer pairs
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Question', 'Generated Answer'])  # Write header
        for generated_question, generated_answer in questions_answers:
            writer.writerow([generated_question, generated_answer])

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_path', default='./path/to/finetuned/model')
    parser.add_argument('--input_csv', default='./path/to/input/csv')
    parser.add_argument('--output_csv', default='./path/to/output/csv')
    parser.add_argument('--delta_min', type=float, default=0.1)
    parser.add_argument('--delta_max', type=float, default=0.5)
    parser.add_argument('--shard_size', type=int, default=20)
    args = parser.parse_args()

    main(args)