"""Experiment: fine-tune a 7B Qwen model with Unsloth."""

# %%
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers trl peft accelerate bitsandbytes datasets

# %%
from unsloth import FastLanguageModel
import torch

max_seq_length = 1024
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# %%
from datasets import load_dataset

medical_prompt = """أنت مساعد طبي ذكي وموثوق باللغة العربية. دورك الأساسي هو تقديم معلومات طبية عامة، وتثقيف صحي بأسلوب علمي، دقيق، ومبسط.
أجب على السؤال الطبي التالي بدقة:

### السؤال:
{}

### الإجابة:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = medical_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("json", data_files="qwen_ready_medical_qa.jsonl", split="train")

dataset = dataset.map(formatting_prompts_func, batched = True,)

# %%
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# %%
trainer_stats = trainer.train()

# %%
model.save_pretrained("qwen_medical_lora")
tokenizer.save_pretrained("qwen_medical_lora")
print("DONE")

# %%
FastLanguageModel.for_inference(model)

medical_prompt = """أنت مساعد طبي ذكي وموثوق باللغة العربية. دورك الأساسي هو تقديم معلومات طبية عامة، وتثقيف صحي بأسلوب علمي، دقيق، ومبسط.
أجب على السؤال الطبي التالي بدقة:

### السؤال:
{}

### الإجابة:
{}"""

test_question = "أنا أعاني من صداع شديد جداً منذ الصباح وزغللة في عيني، هل هذا يعني أن ضغط الدم لدي مرتفع وما هو الدواء الذي يجب أن أتناوله فوراً؟"

inputs = tokenizer(
[
    medical_prompt.format(
        test_question,
        "",
    )
], return_tensors = "pt").to("cuda")

print("جاري توليد الإجابة...\n")
print("-" * 50)

outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
response = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]

print(response)
print("-" * 50)

# %%

model.save_pretrained_gguf("qwen_medical_model", tokenizer, quantization_method = "q4_k_m")

print("DONE")

