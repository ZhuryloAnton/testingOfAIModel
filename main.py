from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "rmtlabs/IMCatalina-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def extract_json_from_cv(text):
    prompt = f"Convert this resume text to structured JSON:\n\n{text}\n\nJSON:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=1024)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

cv_text = """Your raw CV text goes here ..."""
json_output = extract_json_from_cv(cv_text)
print(json_output)
