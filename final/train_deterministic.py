import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import PeftModel

# ================= Configuration =================
BASE_MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"

# Path to your EXISTING Fine-tuned Adapter
# Try relative path first if CWD is /root/Engravers
ADAPTER_PATH = "llama3_gold_quant_checkpoint"
# Fallback to absolute if needed, or check both
ADAPTER_PATH_ABS = "/root/Engravers/llama3_gold_quant_checkpoint"

DATA_FILE = "final/gold_pure_deterministic_train.jsonl"
OUTPUT_DIR = "final/llama3_gold_deterministic_checkpoint"

MAX_SEQ_LENGTH = 2048
DTYPE = None 
LOAD_IN_4BIT = True

def train():
    print(f"🔥 Starting Deterministic Fine-tuning...")
    print(f"   Base Model: {BASE_MODEL_NAME}")
    print(f"   Target Adapter: {ADAPTER_PATH}")
    print(f"   Data:       {DATA_FILE}")
    print(f"   Output:     {OUTPUT_DIR}")
    print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. Load the ORIGINAL Base Model
    print("\n[1/5] Loading Base Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    # 2. Load the Existing Adapter (Checkpoint)
    print(f"\n[2/5] Loading Existing Adapter...")
    
    # Check paths
    final_adapter_path = None
    if os.path.exists(ADAPTER_PATH) and os.path.isdir(ADAPTER_PATH):
        final_adapter_path = ADAPTER_PATH
    elif os.path.exists(ADAPTER_PATH_ABS) and os.path.isdir(ADAPTER_PATH_ABS):
        final_adapter_path = ADAPTER_PATH_ABS
    
    if not final_adapter_path:
        raise FileNotFoundError(f"❌ Could not find adapter directory at '{ADAPTER_PATH}' or '{ADAPTER_PATH_ABS}'")

    print(f"   Found adapter at: {final_adapter_path}")

    # Try loading adapter
    try:
        # Unsloth/Peft load_adapter
        model.load_adapter(final_adapter_path)
        print("   ✅ Adapter loaded successfully via model.load_adapter()!")
    except Exception as e:
        print(f"   ⚠️ model.load_adapter() failed: {e}")
        print("   🔄 Attempting fallback using PeftModel.from_pretrained...")
        try:
            # Fallback: Explicit PeftModel load
            # Note: This wraps the model. Unsloth should handle it, but it might break some Unsloth optimizations if not careful.
            # But usually it's fine for further training.
            model = PeftModel.from_pretrained(model, final_adapter_path, is_trainable=True)
            print("   ✅ Adapter loaded successfully via PeftModel!")
        except Exception as e2:
            raise RuntimeError(f"❌ FATAL: Could not load adapter! Error: {e2}")

    # 3. Configure LoRA for FURTHER Training
    # Important: If we loaded an adapter, we usually just want to enable gradients on it.
    # Unsloth's get_peft_model creates a NEW adapter config if one doesn't exist, or might overwrite?
    # Actually, for continuous training, we often just want to ensure the loaded adapter is trainable.
    
    # However, to be safe and use Unsloth's optimized training loop, we can re-apply the config.
    # The 'get_peft_model' call is generally idempotent or additive.
    
    print("\n[2.5/5] Refreshing LoRA Config for Training...")
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

    # 4. Load & Format Data
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

    # 5. Initialize Trainer
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
            output_dir = OUTPUT_DIR, 
        ),
    )

    # 6. Train
    print("\n[5/5] Training Started 🚀")
    trainer.train()

    print(f"\n✅ Training Complete!")
    
    # Save Model
    print(f"   Saving new checkpoint to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()
