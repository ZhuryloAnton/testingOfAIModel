import json
import re
import torch
import psutil

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===============================
# CONFIG
# ===============================
BASE_MODEL_ID = "microsoft/phi-4"
ADAPTER_ID = "rmtlabs/phi-4-adapter-v1"

DATASET_PATH = "dataset_interim_v6.jsonl"
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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ===============================
# LOAD MODEL
# ===============================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_ID,
    dtype=torch.float16
)


model.eval()
print("✅ Phi-4-Mini + Adapter loaded")

# ===============================
# SAFE JSON EXTRACTION
# ===============================
def extract_last_json(text: str) -> dict:
    matches = re.findall(r"\{[\s\S]*?\}", text)
    if not matches:
        raise ValueError("No JSON found in model output")

    candidate = matches[-1]

    if candidate.count("{") > candidate.count("}"):
        candidate += "}" * (candidate.count("{") - candidate.count("}"))

    return json.loads(candidate)

# ===============================
# CV → STRUCTURED DATA
# ===============================
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

def pretty_print_cv(raw_text: str):
    print("\n" + "=" * 80)
    print("CURRICULUM VITAE")
    print("=" * 80)

    for line in raw_text.splitlines():
        line = line.strip()

        if not line:
            continue

        # Headings
        if line.isupper() or "WORK EXPERIENCE" in line or "EDUCATION" in line:
            print("\n" + line.upper())
            print("-" * len(line))
        else:
            print(line)

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    with open(DATASET_PATH, "r") as f:
        for idx, line in enumerate(f, start=1):
            record = json.loads(line)

            print("\n" + "=" * 60)
            print(f"📄 CV #{idx}")
            print("=" * 60)

            try:
                cv_text = record.get("cv_text", json.dumps(record))
                result = parse_resume(cv_text)

                pretty_print_generated(result)

            except Exception as e:
                print("⚠️ Failed to parse CV")
                print(str(e))
