import json
import pdfplumber
import pytesseract
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "rmtlabs/IMCatalina-v1.0"
PDF_PATH = "cv.pdf"
OUTPUT_JSON = "cv.json"

# ---------------------------
# Load model
# ---------------------------
print("🚀 Loading Catalina model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

# ---------------------------
# Extract text from PDF (with OCR fallback)
# ---------------------------
def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text += page_text + "\n"
            else:
                image = page.to_image(resolution=300).original
                text += pytesseract.image_to_string(image) + "\n"
    return text

# ---------------------------
# Prompt Catalina → JSON
# ---------------------------
def parse_cv_with_ai(cv_text):
    prompt = f"""
You are an AI system that extracts structured data from resumes.

Return ONLY valid JSON.
Do not add explanations.
Do not add markdown.
Start with {{ and end with }}.

Use this exact schema:
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
{cv_text}
\"\"\"
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=700,
        temperature=0.0,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract JSON safely
    start = response.find("{")
    end = response.rfind("}") + 1

    if start == -1 or end == -1:
        raise ValueError("Model did not return JSON")

    return json.loads(response[start:end])

# ---------------------------
# MAIN
# ---------------------------
print("📄 Reading CV...")
cv_text = extract_text_from_pdf(PDF_PATH)

print("🧠 Parsing with AI...")
cv_json = parse_cv_with_ai(cv_text)

with open(OUTPUT_JSON, "w") as f:
    json.dump(cv_json, f, indent=2)

print(f"✅ Done! JSON saved to {OUTPUT_JSON}")
