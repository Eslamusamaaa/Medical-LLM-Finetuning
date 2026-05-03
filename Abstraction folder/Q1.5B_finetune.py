
import os
import json
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)
from trl import SFTTrainer, SFTConfig

MODEL_ID    = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH   = "qwen_ready_medical_qa(2).jsonl"
OUTPUT_DIR  = "outputs/qwen-medical"

SYSTEM_PROMPT = (
    "أنت مساعد طبي ذكي وموثوق. تقدم معلومات وإرشادات طبية عامة باللغة العربية السليمة، "
    "وتوجه المريض دائماً لاستشارة الطبيب في الحالات الحرجة. "
    "لا تقم بوصف جرعات محددة."
)


def load_jsonl(filepath: str) -> list[dict]:
    """Read every line of a JSONL file into a list of dicts."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_to_chatml(example: dict) -> dict:
    
    instruction = example.get("instruction", "").strip()
    user_input  = example.get("input",       "").strip()
    output      = example.get("output",      "").strip()

    user_message = f"{instruction}\n{user_input}" if user_input else instruction

    chatml = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )
    return {"text": chatml}


def build_dataset(filepath: str) -> tuple[Dataset, Dataset]:
    
    raw       = load_jsonl(filepath)
    formatted = [format_to_chatml(r) for r in raw]
    dataset   = Dataset.from_list(formatted)

    split = dataset.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]



print("\n" + "="*60)
print(" Loading tokenizer …")
print("="*60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f" Tokenizer loaded — vocab size: {tokenizer.vocab_size:,}")


print("\n" + "="*60)
print(" Loading base model in BF16 (no quantization) …")
print("="*60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,        # Full BF16 precision
    device_map="auto",                 # Lightning AI: auto-maps to available GPU(s)
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # ~30% faster on L40S
)


model.enable_input_require_grads()
model.gradient_checkpointing_enable()

print(f" Base model loaded")
print(f"   Parameters : {sum(p.numel() for p in model.parameters()):,}")
print(f"   VRAM used  : {torch.cuda.memory_allocated() / 1e9:.2f} GB")


print("\n" + "="*60)
print(" Applying LoRA adapters …")
print("="*60)

lora_config = LoraConfig(
    task_type     = TaskType.CAUSAL_LM,
    r             = 128,
    lora_alpha    = 256,
    lora_dropout  = 0.05,
    bias          = "none",
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
        "gate_proj", "up_proj", "down_proj",        # FFN
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("\n" + "="*60)
print(" Configuring training arguments …")
print("="*60)

sft_config = SFTConfig(
    output_dir = OUTPUT_DIR,
    num_train_epochs = 5,

    per_device_train_batch_size  = 2,
    gradient_accumulation_steps  = 16,   # Effective batch = 32

    learning_rate        = 1e-4,
    lr_scheduler_type    = "cosine",
    warmup_ratio         = 0.05,
    weight_decay         = 0.01,
    optim                = "adamw_torch_fused",  # Fastest on pure BF16

    bf16  = True,
    fp16  = False,

    gradient_checkpointing = True,

    max_seq_length       = 2048,
    dataset_text_field   = "text",  # Column from format_to_chatml()
    packing              =  False,    # Pack short samples for efficiency

    save_strategy        = "epoch",
    save_total_limit     = 5,       # Keep all 5 epoch checkpoints

    eval_strategy           = "no",    # ← وقف الـ evaluation خالص
    load_best_model_at_end  = False,   # ← لازم تتغير معاها  # Keeps the best checkpoint at the end

    logging_dir          = f"{OUTPUT_DIR}/logs",
    logging_steps        = 10,
    report_to            = "none",  # Change to "tensorboard" if desired

    seed                 = 42,

    dataloader_num_workers  = 4,
    dataloader_pin_memory   = True,
    remove_unused_columns   = True,
)


print("\n" + "="*60)
print(" Building dataset …")
print("="*60)

train_dataset, eval_dataset = build_dataset(DATA_PATH)
print(f"Train samples : {len(train_dataset):,}")
print(f"Eval  samples : {len(eval_dataset):,}")

print("\n" + "="*60)
print(" Initialising SFTTrainer …")
print("="*60)

trainer = SFTTrainer(
   
    model             = model,
    args              = sft_config,
    train_dataset     = train_dataset,
    eval_dataset      = eval_dataset,
    processing_class  = tokenizer,   # ← new argument name in TRL 0.12+
)

print("\n" + "="*60)
print("  Starting Training …")
print("="*60)

train_result = trainer.train()

print("\n" + "="*60)
print(" Training Complete — Summary")
print("="*60)
print(f"   Total steps          : {train_result.global_step}")
print(f"   Training loss        : {train_result.training_loss:.4f}")
print(f"   Training time        : {train_result.metrics.get('train_runtime', 0)/60:.1f} min")
print(f"   Samples per second   : {train_result.metrics.get('train_samples_per_second', 0):.1f}")


ADAPTER_DIR = f"{OUTPUT_DIR}/final-adapter"

print("\n" + "="*60)
print(f" Saving raw LoRA adapter → {ADAPTER_DIR}")
print("="*60)

trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print("Raw adapter saved")


MERGED_DIR = f"{OUTPUT_DIR}/merged-model"

print("\n" + "="*60)
print(f" Merging LoRA adapter into base model → {MERGED_DIR}")
print("="*60)

peft_model = trainer.model

print("   Merging weights … (this takes ~1-2 minutes)")
merged_model = peft_model.merge_and_unload()

merged_model.save_pretrained(
    MERGED_DIR,
    safe_serialization = True,   # Save as .safetensors (safer + faster to load)
    max_shard_size     = "2GB",  # Shard large models into manageable files
)
tokenizer.save_pretrained(MERGED_DIR)

print(f" Merged model saved → {MERGED_DIR}")
print(f"   Disk size : ~3 GB (BF16)")
print(f"\n   Directory contents:")
for f in sorted(os.listdir(MERGED_DIR)):
    size = os.path.getsize(os.path.join(MERGED_DIR, f))
    print(f"     {f:45s} {size/1e6:>8.1f} MB")


def run_inference_test(model_dir: str):
    
    print("\n" + "="*60)
    print(" INFERENCE TEST — Merged Model")
    print("="*60)

    print(f"   Loading from : {model_dir}")

    test_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype    = torch.bfloat16,
        device_map     = "auto",
        trust_remote_code = True,
        attn_implementation = "flash_attention_2",
    )
    test_model.eval()

    test_tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code = True,
    )

    print(" Merged model loaded successfully\n")

    test_questions = [
        "ما هي أعراض مرض السكري من النوع الثاني وكيف يمكن الوقاية منه؟",
        "ما الفرق بين الصداع العادي والصداع النصفي؟",
        "متى يجب أن أذهب إلى الطوارئ عند ارتفاع درجة الحرارة؟",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"{'─'*60}")
        print(f"السؤال {i}: {question}")
        print(f"{'─'*60}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]

        prompt_text = test_tokenizer.apply_chat_template(
            messages,
            tokenize              = False,
            add_generation_prompt = True,  # Appends <|im_start|>assistant\n
        )

        inputs = test_tokenizer(
            prompt_text,
            return_tensors = "pt",
        ).to(test_model.device)

        with torch.no_grad():
            output_ids = test_model.generate(
                **inputs,
                max_new_tokens     = 512,
                temperature        = 0.3,   # Low = factual, deterministic
                top_p              = 0.9,
                repetition_penalty = 1.1,   # Prevents phrase repetition
                do_sample          = True,
                eos_token_id       = test_tokenizer.eos_token_id,
                pad_token_id       = test_tokenizer.pad_token_id,
            )

        # Decode only newly generated tokens (exclude the input prompt)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response      = test_tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"الإجابة:\n{response}\n")

    print("="*60)
    print(" Inference test complete!")
    print("="*60)


# Run the test on the freshly merged model
run_inference_test(MERGED_DIR)

