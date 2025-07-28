# ChatPGT - Paul Graham Essay Generation with SFT + DPO

This project implements a complete pipeline for training a Paul Graham essay generator using:
1. **Supervised Fine-Tuning (SFT)** on real Paul Graham essays
2. **Synthetic DPO data generation** using multiple LLMs
3. **Direct Preference Optimization (DPO)** for alignment

## Pipeline Overview

```
SmolLM3-3B → [SFT] → SmolGraham → [DPO Data Gen] → dpo_pairs.csv → [DPO] → SmolGraham-DPO
```

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Set up API keys for DPO data generation:
```bash
export OPENAI_API_KEY="your_openai_key"
export OPENROUTER_API_KEY="your_openrouter_key"  # Optional
```

## Usage

### Step 1: Supervised Fine-Tuning (SFT)

Train the base model on real Paul Graham essays:

```bash
python sft.py
```

This will:
- Load the SmolLM3-3B base model
- Fine-tune on Paul Graham essays from Hugging Face
- Save the SFT model to `SmolGraham/`
- Test with a sample generation

**Output**: `SmolGraham/` directory containing the SFT model

### Step 2: Generate DPO Training Data

Create synthetic preference pairs using multiple LLMs:

```bash
python generate_dpo_data.py
```

This will:
- Load random Wikipedia topics
- Generate essays using multiple models (GPT-3.5, GPT-4, Qwen, Mistral)
- Use GPT-4 to judge which essays are better
- Create preference pairs for DPO training
- Save to `dpo_pairs.csv`

**Output**: `dpo_pairs.csv` with columns:
- `topic`: Essay topic
- `prompt`: Generation prompt
- `chosen`: Preferred essay
- `rejected`: Less preferred essay
- `chosen_model`/`rejected_model`: Source models
- `judgment_reasoning`: Why one was chosen
- `confidence`: Judgment confidence (1-10)

### Step 3: Direct Preference Optimization (DPO)

Further tune the SFT model using preference learning:

```bash
python dpo.py
```

This will:
- Load the SFT model from `SmolGraham/`
- Load preference pairs from `dpo_pairs.csv`
- Filter for high-confidence judgments (confidence ≥ 6)
- Train using DPO to align with preferences
- Save the final model to `SmolGraham-DPO/`
- Test with sample generations

**Output**: `SmolGraham-DPO/` directory containing the final DPO-aligned model

## Model Evolution

1. **Base Model** (`SmolLM3-3B`): General language model
2. **SFT Model** (`SmolGraham`): Specialized for Paul Graham's writing style
3. **DPO Model** (`SmolGraham-DPO`): Aligned to prefer higher-quality outputs

## Configuration

### SFT Training (`sft.py`)
- 2 epochs, batch size 8, learning rate 1e-5
- Max sequence length: 1024 tokens
- Uses real Paul Graham essays from Hugging Face

### DPO Data Generation (`generate_dpo_data.py`)
- Configurable number of topics (default: 10 for testing)
- Multiple LLM providers (OpenAI + OpenRouter)
- Parallel processing with 50 workers
- GPT-4 as preference judge

### DPO Training (`dpo.py`)
- 1 epoch, batch size 2, learning rate 5e-7
- Beta = 0.1 (KL penalty coefficient)
- Filters for confidence ≥ 6 judgments
- Uses sigmoid DPO loss

## Monitoring

Both SFT and DPO training log to TensorBoard:

```bash
tensorboard --logdir=SmolGraham/runs        # SFT metrics
tensorboard --logdir=SmolGraham-DPO/runs    # DPO metrics
```

## Example Usage

After training, use the final model:

```python
from transformers import pipeline, AutoTokenizer

# Load the DPO-trained model
pipe = pipeline("text-generation", model="SmolGraham-DPO")
tokenizer = AutoTokenizer.from_pretrained("SmolGraham-DPO")

# Generate an essay
messages = [{"role": "user", "content": "Write a Paul Graham essay about startup ideas"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

output = pipe(prompt, max_new_tokens=500, temperature=0.7)
print(output[0]["generated_text"])
```

## Notes

- GPU recommended for training (especially DPO with reference model)
- DPO data generation requires API keys and may incur costs
- Start with small datasets for testing (adjust `num_topics` in `generate_dpo_data.py`)
- Models are saved in Hugging Face format and can be uploaded to the Hub
