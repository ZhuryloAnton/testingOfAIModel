from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json

model_name = "rmtlabs/IMCatalina-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
)

# Use a clear instruction
prompt = """
Extract structured CV information from the text below.
Provide output exactly in JSON format with these fields:
{name, age, experience: [{company, years}], skills}
Do not add any extra words.

CV text:
My Name Anton, I have 30 years old experience, I worked in Amazon 5 years, Apple 5 years, Google 10 years, Microsoft 10 years, I like cooking
"""

# Use pipeline for easier handling
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

output = generator(prompt, max_new_tokens=150, do_sample=False)
raw_text = output[0]['generated_text']

print("Raw output:\n", raw_text)

# Try parsing JSON
try:
    json_start = raw_text.find("{")
    json_end = raw_text.rfind("}") + 1
    data_json = raw_text[json_start:json_end]
    data = json.loads(data_json)
    print("\nParsed JSON:\n", data)
except Exception as e:
    print("JSON parsing failed:", e)
