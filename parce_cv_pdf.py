import pdfplumber
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG
# =========================
MODEL_NAME = "rmtlabs/IMCatalina-v1.0"
PDF_FILE = "cv.pdf"

# =========================
# LOAD MODEL
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

# =========================
# HELPERS
# =========================
def clean_text(text: str) -> str:
    # fix duplicated letters from PDF (e.g. DDeevveellooppeerr)
    text = re.sub(r'(.)\1+', r'\1', text)
    # remove weird unicode chars
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return clean_text(text)

def extract_basic_fields(text: str) -> dict:
    email = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    phone = re.search(r'(\+?\d[\d\s\-]{8,}\d)', text)

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    name = lines[0] if lines else ""

    return {
        "name": name,
        "email": email.group(0) if email else "",
        "phone": phone.group(0) if phone else ""
    }

def parse_cv_to_json(cv_text: str) -> str:
    basic = extract_basic_fields(cv_text)

    prompt = f"""
You are a professional resume parser.

Return STRICT JSON ONLY.
Do NOT explain anything.
Do NOT repeat the prompt.

JSON schema:
{{
  "name": "{basic['name']}",
  "email": "{basic['email']}",
  "phone": "{basic['phone']}",
  "address": "",
  "skills": [],
  "experience": [],
  "education": [],
  "languages": []
}}

Resume text:
\"\"\"
{cv_text[:3000]}
\"\"\"

JSON:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=600,
        do_sample=True,
        temperature=0.2,
        eos_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # extract JSON safely
    start = decoded.find("{")
    end = decoded.rfind("}") + 1
    return decoded[start:end]

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("📄 Loading PDF...")
    text = extract_text_from_pdf(PDF_FILE)

    print("🧠 Parsing CV...")
    json_output = parse_cv_to_json(text)

    print("\n======= PARSED JSON =======\n")
    print(json_output)

    # validate JSON
    try:
        parsed = json.loads(json_output)
        print("\n✅ JSON is valid")

        # optional: save to file
        with open("cv.json", "w") as f:
            json.dump(parsed, f, indent=2)
        print("💾 Saved to cv.json")

    except Exception as e:
        print("\n❌ Invalid JSON:", e)
