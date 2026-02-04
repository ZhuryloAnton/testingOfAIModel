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
    prompt = (
        "TASK: Extract structured data from the resume.\n"
        "OUTPUT FORMAT: JSON ONLY.\n"
        "DO NOT add explanations.\n"
        "DO NOT continue the resume.\n\n"
        "FIELDS:\n"
        "- skills (array of strings)\n"
        "- years_of_experience (string)\n"
        "- job_roles (array of strings)\n"
        "- education (array of strings)\n\n"
        "RESUME:\n"
        f"{resume_text}\n\n"
        "JSON:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,                # greedy
            repetition_penalty=1.25,        # 🔥 important
            no_repeat_ngram_size=5,          # 🔥 important
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Hard cut after JSON (extra safety)
    if "}" in text:
        text = text[: text.rfind("}") + 1]

    return text


resume_text = """
Senior Software Engineer with 8+ years of experience.
Expert in Python, PyTorch, NLP, and LLM deployment.
Worked at Google and Amazon.
MSc in Computer Science from Stanford University.
"""

result = analyze_resume(resume_text)
print(result)
