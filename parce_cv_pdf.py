import json
import pdfplumber
import pytesseract
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "rmtlabs/IMCatalina-v1.0"

# ---------------------------
# Load model
# ---------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

# ---------------------------
# Extract text (PDF + OCR)
# ---------------------------
def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                image = page.to_image(resolution=300).original
                text += pytesseract.image_to_string(image) + "\n"
    return text

# ---------------------------
# Ask AI for JSON
# ---------------------------
def cv_to_json(cv_text):
    prompt = f"""
Extract CV data and return ONLY valid JSON.

Schema:
{{
  "name": "",
  "email": "",
  "phone": "",
  "skills": [],
  "experience": [],
  "education": []
}}

CV:
\"\"\"
{cv_text}
\"\"\"
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=600,
        temperature=0.1,
        do_sample=False
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    start = response.find("{")
    end = response.rfind("}") + 1
    return json.loads(response[start:end])

# ---------------------------
# MAIN
# ---------------------------
print("Reading CV...")
text = extract_text("cv.pdf")

print("Parsing with AI...")
result = cv_to_json(text)

with open("cv.json", "w") as f:
    json.dump(result, f, indent=2)

print("Done ✅ → cv.json")
