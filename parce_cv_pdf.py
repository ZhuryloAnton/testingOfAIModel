import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil

# ===============================
# CONFIG
# ===============================
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

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

# ===============================
# LOAD MODEL
# ===============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

model.eval()
print("✅ Model loaded successfully")

# ===============================
# RAW GENERATION (NO MANIPULATION)
# ===============================
def run_raw_generation(user_input: str):
    prompt = user_input.strip()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,          # deterministic
            eos_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text

# ===============================
# TEST
# ===============================
if __name__ == "__main__":
    user_input = """
    put all next information in json format(
40 years, Senior Engineer, java, python, javaScrip, Harvard)
"""

    print("\n🧠 RAW MODEL OUTPUT:")
    print("--------------------------------------------------")
    result = run_raw_generation(user_input)
    print(result)
    print("--------------------------------------------------")
