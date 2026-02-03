import json
import re
import pdfplumber
import pytesseract
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "rmtlabs/IMCatalina-v1.0"
PDF_PATH = "cv.pdf"
OUTPUT_JSON = "cv.json"

print("🚀 Loading Catalina model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

# ---------------------------
# Extract text (PDF + OCR)
# ---------------------------
def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t and t.strip():
                text += t + "\n"
            else:
                img = page.to_image(resolution=300).original
                text += pytesseract.image_to_string(img) + "\n"
    return text[:8000]  # IMPORTANT: limit context

# ---------------------------
# Extract JSON safely
# ---------------------------
def extract_json(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    return match.group(0)

# ---------------------------
# Ask Catalina to COMPLETE JSON
# ---------------------------
def parse_cv_with_ai(cv_text):
    prompt = f"""
Resume:
\"\"\"
{cv_text}
\"\"\"

JSON:
{{
  "name": "
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    json_text = extract_json(response)

    if not json_text:
        print("\n⚠️ Model failed to produce JSON. Raw output:\n")
        print(response)
        return {
            "name": "",
            "email": "",
            "phone": "",
            "address": "",
            "skills": [],
            "experience": [],
            "education": [],
            "languages": []
        }

    try:
        return json.loads(json_text)
    except Exception:
        print("\n⚠️ Invalid JSON produced:\n")
        print(json_text)
        return {
            "name": "",
            "email": "",
            "phone": "",
            "address": "",
            "skills": [],
            "experience": [],
            "education": [],
            "languages": []
        }

# ---------------------------
# MAIN
# ---------------------------
print("📄 Reading CV...")
cv_text = extract_text_from_pdf(PDF_PATH)

print("🧠 Parsing with AI...")
cv_data = parse_cv_with_ai(cv_text)

with open(OUTPUT_JSON, "w") as f:
    json.dump(cv_data, f, indent=2)

print(f"✅ Done → {OUTPUT_JSON}")
