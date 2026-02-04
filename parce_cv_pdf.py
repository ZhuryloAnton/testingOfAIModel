import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil

# ===============================
# CONFIG
# ===============================
MODEL_ID = "rmtlabs/IMCatalina-v1.0"
MAX_NEW_TOKENS = 120   # 🔥 keep small

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
# CATALINA SECTION GENERATOR
# ===============================
def generate_section(resume_text: str, section_name: str):
    """
    section_name examples:
    - PROFESSIONAL SUMMARY
    - KEY SKILLS
    - EXPERIENCE
    """

    prompt = f"""
{section_name}
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
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return only what comes AFTER the section header
    split_key = section_name.upper()
    if split_key in text:
        text = text.split(split_key, 1)[-1]

    return text.strip()

# ===============================
# DEMO
# ===============================
if __name__ == "__main__":

    resume_text = """
Senior Software Engineer with 8+ years of experience.
Expert in Python, PyTorch, NLP, and LLM deployment.
Worked at Google and Amazon.
MSc in Computer Science from Stanford University.
"""

    print("\n🧠 GENERATED PROFESSIONAL SUMMARY")
    print("--------------------------------")
    print(generate_section(resume_text, "PROFESSIONAL SUMMARY"))

    print("\n🧠 GENERATED KEY SKILLS")
    print("--------------------------------")
    print(generate_section(resume_text, "KEY SKILLS"))
