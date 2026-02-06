import json
import re
import torch
import psutil
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===============================
# CONFIG
# ===============================
BASE_MODEL_ID = "microsoft/phi-4-mini"
ADAPTER_ID = "rmtlabs/phi-4-mini-adapter-v1"
MAX_NEW_TOKENS = 512

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
# LOAD MODEL (ONCE)
# ===============================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True
)

model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model.eval()

print("✅ Phi-4 Mini + adapter loaded")

# ===============================
# FASTAPI
# ===============================
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

def extract_last_json(text: str) -> dict:
    matches = re.findall(r"\{[\s\S]*?\}", text)
    if not matches:
        raise ValueError("No JSON found in model output")

    candidate = matches[-1]
    if candidate.count("{") > candidate.count("}"):
        candidate += "}" * (candidate.count("{") - candidate.count("}"))

    return json.loads(candidate)

def parse_resume(resume_text: str) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "Extract structured information from the CV. "
                "Return ONE valid JSON object only."
            )
        },
        {
            "role": "user",
            "content": resume_text
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
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return extract_last_json(decoded)

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    instances = body.get("instances", [])

    predictions = []
    for inst in instances:
        text = inst.get("text", "")
        predictions.append(parse_resume(text))

    return {"predictions": predictions}
