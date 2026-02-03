import pdfplumber
import torch
import json
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG
# =========================
MODEL_NAME = "rmtlabs/IMCatalina-v1.0"
CV_FOLDER = "cvs"
OUTPUT_FOLDER = "output_json"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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
    # fix duplicated letters from PDFs
    text = re.sub(r'(.)\1+', r'\1', text)
    # remove weird unicode symbols
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

def parse_cv_with_ai(cv_text: str) -> str:
    prompt = f"""
You are a professional resume parser.

Analyze the resume and return STRICT JSON ONLY.
Do not explain.
Do not add comments.
Do not repeat the prompt.

JSON schema:
{{
  "name": "",
  "email": "",
  "phone": "",
  "address": "",
  "skills": [],
  "experience": [],
  "education": [],
  "languages": []
}}

Resume:
\"\"\"
{cv_text[:3500]}
\"\"\"

JSON:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=700,
        do_sample=True,
        temperature=0.3,
        eos_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # extract JSON safely
    start = decoded.find("{")
    end = decoded.rfind("}") + 1
    return decoded[start:end]

# =========================
# MAIN LOOP
# =========================
if __name__ == "__main__":
    pdf_files = [f for f in os.listdir(CV_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        print("❌ No PDF files found in cvs/")
        exit(1)

    for pdf_file in pdf_files:
        print(f"\n📄 Processing {pdf_file}...")

        pdf_path = os.path.join(CV_FOLDER, pdf_file)
        text = extract_text_from_pdf(pdf_path)

        json_output = parse_cv_with_ai(text)

        try:
            parsed = json.loads(json_output)

            output_file = os.path.join(
                OUTPUT_FOLDER,
                pdf_file.replace(".pdf", ".json")
            )

            with open(output_file, "w") as f:
                json.dump(parsed, f, indent=2)

            print(f"✅ Saved → {output_file}")

        except Exception as e:
            print(f"❌ Failed to parse {pdf_file}")
            print(e)
