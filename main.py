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
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("sgoel9/paul_graham_essays")['train']

    def build_chat(example: dict, max_len: int = 1024) -> dict:
        messages = [
            {"role": "user", "content": f"Write a Paul Graham essay titled {example['title'][:max_len]}"},
            {"role": "assistant", "content": str(example["text"])}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # Properly truncate
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

        # Decode back to text for SFTTrainer
        truncated_text = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=False)

        return {"text": truncated_text}

    ds = raw.map(
        build_chat,
        remove_columns=raw.column_names,
        fn_kwargs={"max_len": 1024},
    )
    ds = ds.train_test_split(test_size=0.1, seed=42)

    cfg = SFTConfig(
        output_dir="PG_smollm",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_steps=100,
        save_steps=500,
        weight_decay=0.01,
        report_to=["tensorboard"],  # live metrics:  http://localhost:6006
        fp16=True,
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
    out = pipe(
        "Write me a Paul Graham essay about the power of AI",
        max_new_tokens=64
    )[0]["generated_text"]
    print(out)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()
