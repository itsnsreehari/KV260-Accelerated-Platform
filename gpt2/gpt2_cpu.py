import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad_token_id to eos_token_id
tokenizer.pad_token = tokenizer.eos_token

# User inputs the prompt
prompt = input("Enter your prompt: ")

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

# Start timing
start_time = time.time()

# Generate text
output = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=5000,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id
)

# Stop timing
end_time = time.time()

# Decode and print the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated Text:\n", generated_text)

# Compute tokens per second
num_tokens = len(output[0])  # Number of tokens generated
time_taken = end_time - start_time
tok_per_sec = num_tokens / time_taken if time_taken > 0 else 0

print(f"\nPerformance: {tok_per_sec:.2f} tokens/sec")
