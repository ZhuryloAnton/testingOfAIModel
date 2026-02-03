import os
import json
import pdfplumber
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------- CONFIG ----------------
MODEL_NAME = "rmtlabs/IMCatalina-v1.0"
RESUME_FOLDER = "./resumes"  # folder with .pdf or .txt resumes
MAX_TOKENS = 500
TEMPERATURE = 0.2
OUTPUT_FOLDER = "./output"   # where JSON results will be saved
# ----------------------------------------

def load_pdf(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def load_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def load_resume(filepath):
    if filepath.lower().endswith(".pdf"):
        return load_pdf(filepath)
    elif filepath.lower().endswith(".txt"):
        return load_txt(filepath)
    else:
        print(f"Unsupported file type: {filepath}")
        return ""

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    # Process each resume
    for filename in os.listdir(RESUME_FOLDER):
        filepath = os.path.join(RESUME_FOLDER, filename)
        print(f"\nProcessing: {filename}")

        resume_text = load_resume(filepath)
        if not resume_text.strip():
            print("No text found, skipping...")
            continue

        inputs = tokenizer(resume_text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False
        )

        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(result_text)

        # Save JSON output
        json_filename = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(OUTPUT_FOLDER, json_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"resume": resume_text, "ai_output": result_text}, f, indent=2)

        print(f"Saved JSON to: {output_path}")

if __name__ == "__main__":
    main()
