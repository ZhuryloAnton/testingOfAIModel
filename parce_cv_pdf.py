import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil
import json

# ===============================
# CONFIG
# ===============================
MODEL_ID = "rmtlabs/IMCatalina-v1.0"
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
print("Loading Catalina model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

model.eval()
print("✅ Catalina loaded")

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
Use EXACTLY this schema and nothing else:

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

    # 🔑 APPLY CHAT TEMPLATE (THIS IS THE FIX)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

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

    # Try to cut clean JSON
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

    print("\n📄 CATALINA OUTPUT")
    print("-" * 40)
    print(result)

    # Optional: try to parse JSON
    try:
        parsed = json.loads(result)
        print("\n✅ JSON parsed successfully:")
        print(json.dumps(parsed, indent=2))
    except Exception as e:
        print("\n⚠️ Output is not valid JSON yet")
