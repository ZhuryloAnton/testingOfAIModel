import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil
import os

MODEL_ID = "rmtlabs/IMCatalina-v1.0"

def print_system_info():
    print("🖥 System Info")
    print(f"CPU RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print_system_info()

# Enable faster GPU math
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

model.eval()

print("✅ Model loaded successfully")

def analyze_resume(resume_text):
    prompt = f"""
You are an expert resume parser.
Extract structured information from the resume below.

Resume:
{resume_text}

Return the result in JSON with these exact keys:
- skills
- years_of_experience
- job_roles
- education
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

resume_text = """
Senior Software Engineer with 8+ years of experience.
Expert in Python, PyTorch, NLP, and LLM deployment.
Worked at Google and Amazon.
MSc in Computer Science from Stanford University.
"""

result = analyze_resume(resume_text)
print(result)
