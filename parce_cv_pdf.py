import os
import json
import pdfplumber
import torch
import pytesseract
from PIL import Image
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
print("🧠 Loading AI model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# =========================
# PDF + OCR TEXT EXTRACTION
# =========================
def extract_text_with_ocr(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text and len(page_text.strip()) > 30:
                text += page_text + "\n"
            else:
                # OCR fallback
                try:
                    image = page.to_image(resolution=300).original
                    ocr_text = pytesseract.image_to_string(image)
                    text += ocr_text + "\n"
                except Exception as e:
                    print("⚠ OCR failed:", e)
    return text.strip()

# =========================
# AI PARSING
# =========================
def parse_with_ai(cv_text):
    prompt = f"""
You are an AI that extracts structured data from CVs.

Return ONLY valid JSON with this schema:
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

CV TEXT:
\"\"\"
{cv_text}
\"\"\"
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=700,
            temperature=0.2,
            do_sample=False
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Try to extract JSON safely
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        json_text = response[json_start:json_end]
        return json.loads(json_text)
    except Exception:
        print("❌ AI returned invalid JSON")
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

# =========================
# MAIN
# =========================
def main():
    pdfs = [f for f in os.listdir(CV_FOLDER) if f.lower().endswith(".pdf")]

    if not pdfs:
        print("❌ No PDFs found in cvs/")
        return

    for pdf in pdfs:
        pdf_path = os.path.join(CV_FOLDER, pdf)
        print(f"\n📄 Processing {pdf}...")

        text = extract_text_with_ocr(pdf_path)

        if len(text) < 50:
            print("⚠ Very little text extracted")

        data = parse_with_ai(text)

        output_file = os.path.join(
            OUTPUT_FOLDER,
            pdf.replace(".pdf", ".json")
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved → {output_file}")

if __name__ == "__main__":
    main()
