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

# Load model with aggressive GPU usage
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",          # spreads layers safely on GPU
    low_cpu_mem_usage=True
)

model.eval()

print("✅ Model loaded successfully")

def analyze_resume(resume_text, max_tokens=512):
    prompt = f"""
    You are an AI assistant specialized in resume analysis.
    
    Resume:
    {resume_text}
    
    Extract:
    - Key skills
    - Years of experience
    - Job roles
    - Education
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
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9
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