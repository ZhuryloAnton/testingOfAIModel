from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

model_name = "rmtlabs/IMCatalina-v1.0"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Prompt asking for JSON output
prompt = """
Extract the work experience and technology skills from the following CV.
Return the result as JSON with fields: {"name": "", "age": "", "experience": [{"company": "", "years": ""}], "skills": []}.

CV:
My Name Anton, I have 30 years old experience, I worked in Amazon 5 years, Apple 5 years, Google 10 years, Microsoft 10 years, I like cooking
"""

# Tokenize and move to GPU
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate output
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,   # deterministic output
        temperature=0
    )

# Decode output
raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Raw model output:\n", raw_output)

# Attempt to extract JSON
try:
    json_start = raw_output.find("{")
    json_end = raw_output.rfind("}") + 1
    result_json = raw_output[json_start:json_end]
    data = json.loads(result_json)
    print("\nParsed JSON:\n", data)
except Exception as e:
    print("Failed to parse JSON:", e)
