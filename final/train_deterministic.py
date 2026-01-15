import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ================= Configuration =================
# Absolute paths based on your environment
BASE_MODEL_PATH = "/root/Engravers/llama3_gold_quant_checkpoint" # Start from previous fine-tune
DATA_FILE = "final/gold_pure_deterministic_train.jsonl"
OUTPUT_DIR = "final/llama3_gold_deterministic_checkpoint"

MAX_SEQ_LENGTH = 2048
DTYPE = None # Auto-detect
LOAD_IN_4BIT = True

def train():
    print(f"🔥 Starting Deterministic Fine-tuning...")
    print(f"   Base Model: {BASE_MODEL_PATH}")
    print(f"   Data: {DATA_FILE}")
    print(f"   Output: {OUTPUT_DIR}")
    
    # Safety Check: Ensure we aren't overwriting the source
    if os.path.abspath(BASE_MODEL_PATH) == os.path.abspath(OUTPUT_DIR):
        raise ValueError("Output directory cannot be the same as the base model path! This would overwrite your checkpoint.")

    print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. Load the Pre-trained (Fine-tuned) Model
    # Unsloth will automatically load the base model (llama-3-8b) + the adapter at this path
    print("\n[1/5] Loading Model from Checkpoint...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL_PATH,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    # 2. Configure LoRA for *Further* Training
    # Since we loaded an adapter, we need to ensure it's trainable or add new adapters.
    # Calling get_peft_model on a model that already has PEFT might be tricky.
    # However, Unsloth handles 'continue training' well.
    print("\n[2/5] Configuring LoRA Adapters (Continuing Training)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 64, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )

    # 3. Load & Format Data
    print("\n[3/5] Loading & Formatting Data...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Training data file not found: {DATA_FILE}")
        
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token 
    
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True)
    print(f"   Dataset size: {len(dataset)} samples")

    # 4. Initialize Trainer
    print("\n[4/5] Initializing Trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 4,
        packing = False, 
        args = TrainingArguments(
            per_device_train_batch_size = 16, 
            gradient_accumulation_steps = 1,
            warmup_steps = 10,
            max_steps = 250, 
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = OUTPUT_DIR, # Saving to NEW directory
        ),
    )

    # 5. Train
    print("\n[5/5] Training Started 🚀")
    trainer.train()

    print(f"\n✅ Training Complete!")
    
    # Save Model
    print(f"   Saving new checkpoint to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()
