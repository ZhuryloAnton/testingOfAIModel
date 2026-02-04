import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil
import re

# ===============================
# CONFIG
# ===============================
MODEL_ID = "rmtlabs/IMCatalina-v1.0"
MAX_NEW_TOKENS = 80   # 🔥 very important

BAD_STOP_WORDS = [
    "Date",
    "CAREIENCE",
    "CURER",
    "Father",
    "Address",
    "Synopsis",
    "Visit",
    "Electrical",
    "JOB",
]

# ===============================
# SYSTEM INFO
# ===============================
def print_system_info():
    print("🖥 System Info")
    print(f"CPU RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print_system_info()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ===============================
# LOAD MODEL
# ===============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype=torch.float16,
    low_cpu_mem_usage=True
)

model.eval()
print("✅ Model loaded successfully")

# ===============================
# CLEAN GENERATION
# ===============================
def generate_section(resume_text: str, section_name: str) -> str:
    prompt = f"""{section_name}
{resume_text.strip()}

{section_name}
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
            max_new_tokens=80,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # 1️⃣ Remove everything up to the SECOND section header
    parts = text.split(section_name)
    if len(parts) >= 3:
        text = parts[2]
    elif len(parts) == 2:
        text = parts[1]
    else:
        text = ""

    # 2️⃣ Remove echoed resume lines
    for line in resume_text.strip().splitlines():
        text = text.replace(line.strip(), "")

    # 3️⃣ Stop on garbage signals
    BAD_STOP_WORDS = [
        "Date", "CARE", "CURER", "Father", "Address",
        "Synopsis", "Electrical", "JOB"
    ]

    clean_lines = []
    for line in text.splitlines():
        if any(bad.lower() in line.lower() for bad in BAD_STOP_WORDS):
            break
        clean_lines.append(line)

    cleaned = "\n".join(clean_lines).strip()

    # 4️⃣ Final normalization
    cleaned = cleaned.replace(section_name, "").strip()
    cleaned = " ".join(cleaned.split())

    return cleaned


# ===============================
# DEMO
# ===============================
if __name__ == "__main__":

    resume_text = """
Senior Software Engineer with 8+ years of experience banana banana banana banana banana.
Expert in Python, PyTorch, NLP, and LLM deployment.
Worked at Google and Amazon.
MSc in Computer Science from Stanford University.
"""

    print("\n🧠 CLEAN PROFESSIONAL SUMMARY")
    print("--------------------------------")
    print(generate_section(resume_text, "PROFESSIONAL SUMMARY"))

    print("\n🧠 CLEAN KEY SKILLS")
    print("--------------------------------")
    print(generate_section(resume_text, "KEY SKILLS"))
