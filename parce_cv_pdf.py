import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil
import re
import json

# ===============================
# CONFIG
# ===============================
MODEL_ID = "rmtlabs/IMCatalina-v1.0"
MAX_INPUT_LEN = 4096
MAX_NEW_TOKENS = 200

# ===============================
# SYSTEM INFO
# ===============================
def print_system_info():
    print("🖥 System Info")
    print(f"CPU RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print_system_info()

# Enable fast math
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ===============================
# LOAD MODEL
# ===============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

model.eval()
print("✅ Model loaded successfully")

# ===============================
# STAGE 1: MODEL EXTRACTION
# ===============================
def extract_resume_bullets(resume_text: str) -> str:
    prompt = (
        "Extract information from the resume.\n"
        "IMPORTANT RULES:\n"
        "- Use bullet points starting with '- '\n"
        "- If information is missing, leave the section empty\n"
        "- Do NOT repeat section titles as content\n\n"
        "Skills:\n"
        "- \n"
        "Experience:\n"
        "- \n"
        "Job Titles:\n"
        "- \n"
        "Education:\n"
        "- \n\n"
        "Resume:\n"
        f"{resume_text}\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
        add_special_tokens=True
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=5,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# ===============================
# STAGE 2: BULLETS → JSON
# ===============================
def bullets_to_json(text: str) -> dict:
    SECTION_NAMES = {"skills", "experience", "job titles", "education", "resume"}

    def extract(section):
        pattern = rf"{section}:\n(.*?)(\n\n|$)"
        match = re.search(pattern, text, re.S | re.I)
        if not match:
            return []

        items = []
        for line in match.group(1).splitlines():
            line = line.lstrip("-• ").strip()
            if not line:
                continue
            if line.lower().rstrip(":") in SECTION_NAMES:
                continue
            items.append(line)
        return items

    experience = extract("Experience")

    return {
        "skills": extract("Skills"),
        "years_of_experience": experience[0] if experience else "",
        "job_roles": extract("Job Titles"),
        "education": extract("Education")
    }

def normalize_resume_text(text: str) -> str:
    """
    Convert short / keyword-style input into resume-like sentences
    so resume LLMs can understand it.
    """
    text = text.strip()

    # If it's very short or comma-separated, expand it
    if len(text.split()) < 25 or "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]

        sentences = []
        skills = []
        education = []
        roles = []
        experience = []

        for p in parts:
            lower = p.lower()
            if "year" in lower:
                experience.append(p)
            elif any(k in lower for k in ["engineer", "developer", "manager"]):
                roles.append(p)
            elif any(k in lower for k in ["harvard", "university", "college", "school"]):
                education.append(p)
            else:
                skills.append(p)

        if roles:
            sentences.append(f"Worked as {', '.join(roles)}.")
        if experience:
            sentences.append(f"Has {', '.join(experience)} of professional experience.")
        if skills:
            sentences.append(f"Skilled in {', '.join(skills)}.")
        if education:
            sentences.append(f"Education includes {', '.join(education)}.")

        return " ".join(sentences)

    return text
# ===============================
# PUBLIC API
# ===============================
def parse_resume(resume_text: str) -> dict:
    resume_text = normalize_resume_text(resume_text)
    raw_output = extract_resume_bullets(resume_text)
    structured = bullets_to_json(raw_output)
    return structured

# ===============================
# TEST
# ===============================
if __name__ == "__main__":
    resume_text = "40 years, Senior Engineer, java, python, javaScrip, Harvard"

    result = parse_resume(resume_text)
    print("\n📄 Parsed Resume JSON:")
    print(json.dumps(result, indent=2))
