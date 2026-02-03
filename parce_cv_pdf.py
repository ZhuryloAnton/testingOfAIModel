import pdfplumber
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "rmtlabs/IMCatalina-v1.0"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def parse_cv_to_json(cv_text):
    prompt = f"""
You are an AI that extracts structured data from resumes.

Return ONLY valid JSON in this format:
{{
  "name": "",
  "email": "",
  "phone": "",
  "skills": [],
  "experience": [],
  "education": []
}}

Resume text:
{cv_text}
JSON:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=800,
        temperature=0.1
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

if __name__ == "__main__":
    pdf_file = "ITResumePoliskoAnton.pdf"  # <-- your PDF file
    text = extract_text_from_pdf(pdf_file)
    json_output = parse_cv_to_json(text)

    print("======= RAW MODEL OUTPUT =======")
    print(json_output)
