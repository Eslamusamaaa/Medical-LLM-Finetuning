import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

print("Starting Local Smoke Test...")

data_path = "data/chatml_dataset.csv" 
dataset = load_dataset("csv", data_files=data_path, split="train[:10]") 
print("Dataset loaded successfully.")

model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
print("Model loaded successfully.")


peft_config = LoraConfig(
    r=8, 
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir="./smoke_test_results",
    per_device_train_batch_size=1,
    max_steps=3,
    logging_steps=1,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=128,
    tokenizer=tokenizer,
    args=training_args,
)

print("Running training steps...")
trainer.train()

print("Smoke test passed. Pipeline is ready.")