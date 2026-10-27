"""Experiment: fine-tune Qwen2.5-0.5B (smaller baseline)."""

# %%
# !pip install -q transformers datasets peft accelerate bitsandbytes trl sentencepiece

# %%
# !pip install -q "trl==0.8.6" "transformers==4.40.0" "peft==0.10.0"

# %%
import torch

print(f"CUDA available : {torch.cuda.is_available()}")
print(f"GPU count      : {torch.cuda.device_count()}")
print(f"GPU name       : {torch.cuda.get_device_name(0)}")
print(f"Torch version  : {torch.__version__}")

# %%
import pandas as pd
from datasets import Dataset

# Load our pre-processed ChatML dataset
df = pd.read_csv('/kaggle/input/datasets/hagargalall/chatml-dataset-csv/chatml_dataset.csv')

print(f"Total rows : {len(df)}")
print(f"Columns    : {df.columns.tolist()}")
print("\nSample:")
print(df['text'].iloc[0][:300])

# %%
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Split 95/5
train_df, eval_df = train_test_split(df, test_size=0.05, random_state=42)

# Convert to HuggingFace Dataset
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
eval_ds  = Dataset.from_pandas(eval_df.reset_index(drop=True))

print(f"Train samples      : {len(train_ds)}")
print(f"Validation samples : {len(eval_ds)}")

# %%
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Tokenizer loaded  ")
print(f"Vocab size        : {tokenizer.vocab_size}")
print(f"Pad token         : {tokenizer.pad_token}")

# %%
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False
print("Model loaded in 4-bit ")
print(f"Parameters: {model.num_parameters():,}")

# %%
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %%
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="/kaggle/working/qwen-medical-lora",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=True,
    logging_steps=25,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    optim="paged_adamw_8bit",
)

print("Training arguments set")

# %%
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,
)

print("Starting training")
trainer.train()
print("Training complete!")

# %%
import os

path = "/kaggle/working/qwen-medical-lora"

if os.path.exists(path):
    print("Files found:")
    for f in os.listdir(path):
        print(f"  {f}")
else:
    print(" No saved files found — need to retrain")

# %%
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

MODEL_ID    = "Qwen/Qwen2.5-0.5B-Instruct"
CHECKPOINT  = "/kaggle/working/qwen-medical-lora/checkpoint-5800"
MERGED_DIR  = "/kaggle/working/qwen-medical-merged"

print("Loading base model for merging...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)

print("Loading LoRA adapter from checkpoint...")
model = PeftModel.from_pretrained(base_model, CHECKPOINT)

print("Merging weights...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(MERGED_DIR, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.save_pretrained(MERGED_DIR)

print("Merged model saved to:", MERGED_DIR)

# %%
import shutil

print("Zipping merged model...")
shutil.make_archive(
    '/kaggle/working/qwen-medical-merged',
    'zip',
    '/kaggle/working/qwen-medical-merged'
)
print("Zip created!")

# Check zip size
import os
size = os.path.getsize('/kaggle/working/qwen-medical-merged.zip')
print(f"Zip size: {size / (1024*1024*1024):.2f} GB")

# %%
import os
os.makedirs('/kaggle/working/output', exist_ok=True)

shutil.copy(
    '/kaggle/working/qwen-medical-merged.zip',
    '/kaggle/working/output/qwen-medical-merged.zip'
)
print("Done")

# %%
import os

# Create a dataset output folder
os.makedirs('/kaggle/working/dataset_output', exist_ok=True)

# Copy zip there
import shutil
shutil.copy(
    '/kaggle/working/qwen-medical-merged.zip',
    '/kaggle/working/dataset_output/qwen-medical-merged.zip'
)
print("Done")

