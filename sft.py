import os
import random
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from trl import SFTTrainer, SFTConfig


def train(hub_model_name: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "HuggingFaceTB/SmolLM3-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dropout_prob = 0.1
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_prob
    raw = load_dataset("sgoel9/paul_graham_essays")['train']

    def build_chat(example: dict) -> dict:
        prompts = [
            f"Write a Paul Graham essay titled {example['title']}",
            f"Create an essay in Paul Graham's style about {example['title']}",
            f"Discuss {example['title']} in the manner of Paul Graham"
        ]
        prompt = random.choice(prompts)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": str(example["text"])}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = raw.map(
        build_chat,
        remove_columns=raw.column_names,
    )
    ds = ds.train_test_split(test_size=0.2, seed=42)

    cfg = SFTConfig(
        output_dir="SmolGraham",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_steps=100,
        save_steps=2000,
        weight_decay=0.005,
        report_to=["tensorboard"],  # live metrics:  http://localhost:6006
        fp16=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        dataset_text_field="text",
        max_grad_norm=1.0,
        packing=False,
        # Add hub integration
        push_to_hub=True if hub_model_name else False,
        hub_model_id=hub_model_name if hub_model_name else None,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        args=cfg,
        processing_class=tokenizer
    )

    if device == "cuda":
        torch.cuda.empty_cache()

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    
    # Push to hub if specified
    if hub_model_name:
        print(f"🚀 Pushing model to Hugging Face Hub: {hub_model_name}")
        try:
            # Push model (token is automatically retrieved from HF_TOKEN env var or ~/.cache/huggingface/token)
            trainer.model.push_to_hub(hub_model_name)
            # Push tokenizer
            tokenizer.push_to_hub(hub_model_name)
            print(f"✅ Model successfully pushed to {hub_model_name}")
        except Exception as e:
            print(f"❌ Failed to push to hub: {e}")
            print("💡 Make sure you're logged in with `huggingface-cli login`")
            print("💡 Or set the HF_TOKEN environment variable with your token")

    pipe = pipeline(
        "text-generation",
        model=cfg.output_dir,
        device=0 if device == "cuda" else "cpu"
    )
    messages = [
        {"role": "user",
         "content": "Write a Paul Graham essay about the power of AI"}
    ]
    # Load the saved tokenizer to ensure consistency with the trained model
    saved_tokenizer = AutoTokenizer.from_pretrained(cfg.output_dir)
    prompt = saved_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # tells the model "your turn next"
    )
    out = pipe(
        prompt,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )[0]["generated_text"]
    print(out)
    
    return hub_model_name if hub_model_name else cfg.output_dir


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # You can specify a hub model name here, e.g., "your-username/SmolGraham"
    # If None, will only save locally
    hub_name = os.getenv("HF_MODEL_NAME", "SmolGraham-SFT")  # Default hub name
    train(hub_name)
