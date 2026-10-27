"""Compare answers of the base model and the fine-tuned model."""

# %%
# !pip install -q transformers accelerate torch

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

system_prompt = (
    "أنت مساعد طبي ذكي وموثوق. تقدم معلومات وإرشادات طبية عامة باللغة العربية السليمة، "
    "وتوجه المريض دائماً لاستشارة الطبيب في الحالات الحرجة. "
    "لا تقم بوصف جرعات محددة."
)
user_query = "انا عاني من صداع شديد منذ الصباح وزغللة في العين , ما هو الوداء المناسب لذلك"

def generate_response(model, tokenizer, prompt, user_input):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True
    )

    response_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]

# %%
base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print("Base Model...")
tokenizer_base = AutoTokenizer.from_pretrained(base_model_id)
model_base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype="auto", device_map="auto")

print("\n Base Model answer")
print(generate_response(model_base, tokenizer_base, system_prompt, user_query))

del model_base
torch.cuda.empty_cache()

# %%
merged_model_id = "hagora-30/qwen2.5-1.5B-medical-arabic"

print("Merged Model from Hugging Face...")
tokenizer_merged = AutoTokenizer.from_pretrained(merged_model_id)
model_merged = AutoModelForCausalLM.from_pretrained(merged_model_id, torch_dtype="auto", device_map="auto")

print("\n Merged Model ")
print(generate_response(model_merged, tokenizer_merged, system_prompt, user_query))

