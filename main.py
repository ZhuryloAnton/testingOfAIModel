from transformers import AutoTokenizer, AutoModelForCausalLM
import pdfplumber
import torch

MODEL_NAME = "rmtlabs/IMCatalina-v1.0"
RESUME_FILE = "resume.pdf"  # your resume file in PDF
MAX_TOKENS = 500
TEMPERATURE = 0.2

def load_resume():
    text=""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text += text + "\n"
    return text

def main():
    print("I am loading my LORD")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    resume_text = load_resume(RESUME_FILE)
    print("I teared out the information :>")
    inputs = tokenizer(resume_text, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        do_sample=False
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print("output")
    print(result)

if __name__ == "__main__":
    main()