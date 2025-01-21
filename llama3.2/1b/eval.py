# import torch
# from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Environment setup
# os.environ["HF_ALLOW_CODE_EVAL"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Load model and tokenizer
# access_token = os.getenv("HF_ACCESS_TOKEN")
# model_id = "meta-llama/Llama-3.2-1B-Instruct"
# cache_dir = "../../models_weights/"

# model_id = "meta-llama/Llama-3.2-1B-Instruct"
# cache_dir = "../../models_weights/"

# # Load the model and tokenizer with cache_dir
# model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, token=access_token)
# tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, token=access_token)

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

# messages = [
#     {"role": "system", "content": "You are expert in deep learning architectures, optimizations, models, and math at a professor level. You can define novel unpublished architectures, and optimizations, for a task when asked. Your response will be methodological like a research paper in AI confernces. "},
#     # {"role": "user", "content": "Who are you?"},
#     {"role": "user", "content": "define a new norm layer that combines RMSNorm with batchnorm, that is only variance computation is replaced with RMS and mean shifting is done as in batchnorm. Explain what benefit it can have compared to batchnorm in terms of computational efficiency"},
# ]

# outputs = pipe(
#     messages,
#     max_new_tokens=1e6,
# )

# print((outputs[0]['generated_text'][-1]['content']))

import torch, json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment setup
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load model and tokenizer
access_token = os.getenv("HF_ACCESS_TOKEN")
model_id = "meta-llama/Llama-3.2-1B-Instruct"
cache_dir = "../../models_weights/"

model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, token=access_token)
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, token=access_token)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Ensure tokenizer has special tokens
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': '</s>'})
if len(tokenizer) > model.config.vocab_size:
    model.resize_token_embeddings(len(tokenizer))

# Load GSM8K dataset
gsm8k = load_dataset("gsm8k", "main")['test']

# Chain of Thought instruction
cot_instruction = (
    "Solve the following math problem step by step. Provide a detailed explanation, "
    "and end with the answer on a new line in the format 'Answer: <final_answer>'."
)

# Function to generate CoT answer
def generate_cot_answer(prompt):
    full_prompt = cot_instruction + "\n" + prompt
    inputs = tokenizer(full_prompt, return_tensors="pt").to(pipe.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=512,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Evaluate on GSM8K
correct_count = 0
total_count = len(gsm8k)
results = []

for problem in gsm8k:
    problem_prompt = problem['question']
    ground_truth = problem['answer']
    generated_answer = generate_cot_answer(problem_prompt)

    if "Answer:" in generated_answer:
        final_answer = generated_answer.split("Answer:")[-1].strip()
    else:
        final_answer = None

    if final_answer == ground_truth:
        correct_count += 1

    results.append({
        "prompt": problem_prompt,
        "ground_truth": ground_truth,
        "generated_answer": generated_answer,
    })
    break

# Calculate accuracy
accuracy = correct_count / total_count
print(f"Accuracy on GSM8K: {accuracy * 100:.2f}%")

# Save results
with open("gsm8k_results.json", "w") as f:
    json.dump(results, f, indent=4)