import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil

# ===============================
# CONFIG
# ===============================
MODEL_ID = "https://huggingface.co/rmtlabs/IMCatalina-v1.0"

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
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

model.eval()
print("✅ Model loaded successfully")

# ===============================
# RAW GENERATION TEST
# ===============================
def run_model(resume_text: str):
    messages = [
        {
            "role": "system",
            "content": "You are an expert resume parser."
        },
        {
            "role": "user",
            "content": f"""
Extract the following fields from the resume and return STRICT JSON:

- skills (array)
- years_of_experience (string)
- job_roles (array)
- education (array)

Resume:
{resume_text}

Return JSON only.
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
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.1
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n🧠 RAW MODEL OUTPUT:\n")
    print(text)

# ===============================
# TEST
# ===============================
if __name__ == "__main__":
    resume = """
Senior Software Engineer with 20+ years of experience.
Expert in Python, PyTorch, NLP, and LLM deployment, also Java and Kafka.
Worked at Google, Amazon and Apple.
MSc in Data Science from Stanford University.
"""
    run_model(resume)
