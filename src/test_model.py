import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH  = r"C:\Users\Lenovo\Medical-LLM-Finetuning\deployment\models\7b_medical_model.gguf"
OUTPUT_FILE = r"C:\Users\Lenovo\Medical-LLM-Finetuning\reports\test_output.txt"

SYSTEM_PROMPT = """أنت مساعد طبي ذكي وموثوق باللغة العربية. دورك الأساسي هو تقديم معلومات طبية عامة، وتثقيف صحي بأسلوب علمي، دقيق، ومبسط.
يجب أن تكون إجابتك مفصلة، ومنسقة في نقاط مرتبة (Bullet points) لتسهيل القراءة.
أجب على السؤال الطبي التالي بدقة:

### السؤال:
{}

### الإجابة:
"""

class Tee:
    def __init__(self, file):
        self.file    = file
        self.terminal = sys.stdout
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    def flush(self):
        self.terminal.flush()
        self.file.flush()

f          = open(OUTPUT_FILE, 'w', encoding='utf-8')
sys.stdout = Tee(f)

print("Loading model... (this may take a minute)")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()
print("Model loaded\n")

def ask(question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    text   = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

questions = [
    "ماهي مميزات وعيوب الدواء جلوكوفانس ؟",
    "كيف أتعامل مع ارتفاع ضغط الدم؟",
    "ما الفرق بين الإنفلونزا ونزلة البرد؟",
]

for q in questions:
    print(f"السؤال: {q}")
    print(f"الجواب: {ask(q)}")


print("\nTest complete! Results saved to test_output.txt")
f.close()