import json
import re
import pdfplumber
import pytesseract
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "rmtlabs/IMCatalina-v1.0"
PDF_PATH = "cv.pdf"
OUTPUT_JSON = "cv.json"

# --------------------------------------------------
# Load model (no unsupported generation flags)
# --------------------------------------------------
print("🚀 Loading Catalina model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

# --------------------------------------------------
# Extract text from PDF (text + OCR fallback)
# --------------------------------------------------
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

# --------------------------------------------------
# Extract first JSON object from model output
# --------------------------------------------------
def extract_json_from_text(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("❌ No JSON object found in model output")
    return match.group(0)

# --------------------------------------------------
# Ask Catalina to parse CV
# --------------------------------------------------
def parse_cv_with_ai(cv_text):
    prompt = f"""
You are an AI that extracts structured data from resumes.

Return ONLY valid JSON.
No explanations.
No markdown.
Start with {{ and end with }}.

Schema:
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
        max_new_tokens=600,
        do_sample=False   # IMPORTANT: works with Catalina
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    try:
        json_text = extract_json_from_text(response)
        return json.loads(json_text)
    except Exception:
        print("\n⚠️ Model output (debug):\n")
        print(response)
        raise

# --------------------------------------------------
# MAIN
# --------------------------------------------------
print("📄 Reading CV...")
cv_text = extract_text_from_pdf(PDF_PATH)

print("🧠 Parsing with AI...")
cv_data = parse_cv_with_ai(cv_text)

with open(OUTPUT_JSON, "w") as f:
    json.dump(cv_data, f, indent=2)

print(f"✅ Done! JSON saved to {OUTPUT_JSON}")
