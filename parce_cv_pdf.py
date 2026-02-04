import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# CONFIG
# ----------------------------
MODEL_ID = "rmtlabs/IMCatalina-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ----------------------------
# LOAD MODEL
# ----------------------------
print("Loading Catalina model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=DTYPE,
)

model.eval()
print("✅ Catalina loaded\n")

# ----------------------------
# MESSY OCR-LIKE INPUT
# ----------------------------
raw_resume_text = """
Experince experince experince
Senior softwere enginer
Work at Goog 2018 - 2022 develop NLP systm pytorch torch
MSc computer scince Stand ford universty
"""

# ----------------------------
# PROMPT (VERY IMPORTANT)
# ----------------------------
prompt = f"""
Rewrite the following resume text so it is clean, readable, and professional.
Do NOT add new information.
Do NOT invent skills.
Only fix grammar, repetition, and formatting.

Text:
{raw_resume_text}

Clean version:
"""

# ----------------------------
# GENERATE CLEAN TEXT
# ----------------------------
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,          # deterministic
        temperature=0.0,          # conservative
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
    )

decoded = tokenizer.decode(output[0], skip_special_tokens=True)

# ----------------------------
# POST-PROCESS
# ----------------------------
clean_text = decoded.split("Clean version:")[-1].strip()

print("🧠 CLEANED RESUME TEXT")
print("-" * 40)
print(clean_text)
