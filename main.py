from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "rmtlabs/IMCatalina-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = ("Extract the job experience and technology skills from this CV "
          "My Name Anton, I have 30 years old experience, I worked in Amazon 5 years, Apple 5 years, Google 10 years, Mirosoft 10 years, I like cooking ")


inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
