import pdfplumber
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "rmtlabs/IMCatalina-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

def clean_text(text: str) -> str:
    # remove duplicated characters like "DDiiggiittaalliizzeedd"
    text = re.sub(r'(.)\1+', r'\1', text)
    # remove strange symbols
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return clean_text(text)

def parse_cv_to_json(cv_text):
    basic = extract_basic_fields(cv_text)

    prompt = f"""
You are a resume parser.
Return STRICT JSON ONLY.

If a field is already provided, reuse it.

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

Resume:
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

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # extract JSON safely
    json_start = result.find("{")
    json_end = result.rfind("}") + 1
    return result[json_start:json_end]

if __name__ == "__main__":
    pdf_file = "cv.pdf"
    text = extract_text_from_pdf(pdf_file)
    json_output = parse_cv_to_json(text)

    print("\n======= PARSED JSON =======\n")
    print(json_output)

    # optional: validate JSON
    try:
        parsed = json.loads(json_output)
        print("\n✅ JSON is valid")
    except Exception as e:
        print("\n❌ Invalid JSON:", e)
