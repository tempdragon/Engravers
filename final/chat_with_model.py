import torch
from unsloth import FastLanguageModel
import os
import argparse

# Force offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

FINANCE_MODEL_PATH = "/home/dragon/AI/llama-3-8B-4bit-finance"
BASE_MODEL_PATH = (
    "~/.cache/modelscope/hub/models/unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
)


def chat_loop():
    print("Select Model to Chat With:")
    print(f"1. Finance Model ({FINANCE_MODEL_PATH})")
    print(f"2. Base Instruct Model ({BASE_MODEL_PATH})")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "2":
        model_path = BASE_MODEL_PATH
        print("Selected: Base Instruct Model")
    else:
        model_path = FINANCE_MODEL_PATH
        print("Selected: Finance Model")

    print(f"Loading Model from {model_path}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
            local_files_only=True,
        )
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print(
            "Note: If 'local_files_only=True' failed, the model might not be fully cached or 'config.json' is missing."
        )
        print(
            "Attempting to load without 'local_files_only=True' (requires internet if not cached)..."
        )
        # Temporarily enable online mode just for this retry if needed, but keeping offline env var might conflict.
        # Let's just try loading without the flag, hoping it finds the cache properly.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
            # local_files_only=False # Let library decide
        )

    FastLanguageModel.for_inference(model)
    print("\n✅ Model Loaded! You can now chat with the FinLLM.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("\n[User]: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            # Use the Alpaca format which the model was likely trained on or expects
            prompt = f"""### Instruction:
You are a financial AI assistant. Answer the following question.

### Input:
{user_input}

### Response:
"""
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                use_cache=True,
                temperature=0.7,  # Add some creativity for chat
                do_sample=True,
            )
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

            # Extract just the response part
            response_text = response.split("### Response:")[-1].strip()

            print(f"\n[FinLLM]: {response_text}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    chat_loop()
