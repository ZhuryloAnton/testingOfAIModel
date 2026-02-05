import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import psutil
import json


# ===============================
# CONFIG
# ===============================
BASE_MODEL_ID = "microsoft/phi-4-mini-instruct"
ADAPTER_ID = "rmtlabs/phi-4-mini-adapter-v1"
MAX_NEW_TOKENS = 256


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

# Faster GPU math
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ===============================
# LOAD MODEL
# ===============================
print("Loading Phi-4 Mini base model...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

print("Loading adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_ID,
    torch_dtype=torch.float16
)

model.eval()
print("✅ Phi-4-Mini + Adapter loaded")


# ===============================
# RESUME → JSON
# ===============================
def parse_resume(resume_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an AI that converts resumes into structured JSON. Output JSON only."
        },
        {
            "role": "user",
            "content": f"""
Convert the following CV into JSON.
Fill this schema using the CV content.
If a field is clearly present, extract it.
Do not invent information.

{{
  "skills": [],
  "years_of_experience": "",
  "job_roles": [],
  "education": []
}}

CV:
{resume_text}
"""
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    if "{" in decoded and "}" in decoded:
        decoded = decoded[decoded.find("{"): decoded.rfind("}") + 1]

    return decoded


# ===============================
# TEST
# ===============================
if __name__ == "__main__":
    resume_text = """
Senior Software Engineer with 8+ years of experience.
Expert in Python, PyTorch, NLP, and LLM deployment.
Worked at Google and Amazon.
MSc in Computer Science from Stanford University.
"""

    result = parse_resume(resume_text)

    print("\n📄 PHI-4-MINI ADAPTER OUTPUT")
    print("-" * 40)
    print(result)

    try:
        parsed = json.loads(result)
        print("\n✅ JSON parsed successfully:")
        print(json.dumps(parsed, indent=2))
    except Exception:
        print("\n⚠️ Output is not valid JSON yet")

