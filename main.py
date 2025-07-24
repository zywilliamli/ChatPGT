import os

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from trl import SFTTrainer, SFTConfig


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("sgoel9/paul_graham_essays")['train']

    def build_chat(example: dict) -> dict:
        messages = [
            {"role": "user", "content": f"Write a Paul Graham essay titled {example['title']}"},
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
        output_dir="PG_smollm",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_steps=1,
        save_steps=500,
        weight_decay=0.005,
        report_to=["tensorboard"],  # live metrics:  http://localhost:6006
        fp16=False,
        bf16=False,
        packing=False,  # set True if packing multiple msgs
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        max_seq_length=1024,
        dataset_text_field="text",
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

    pipe = pipeline(
        "text-generation",
        model=cfg.output_dir,
        device=0 if device == "cuda" else "cpu"
    )
    messages = [
        {"role": "user",
         "content": "Write a Paul Graham essay about the power of AI"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # tells the model “your turn next”
    )
    out = pipe(
        prompt,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )[0]["generated_text"]
    print(out)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()
