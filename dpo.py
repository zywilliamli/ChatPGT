import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from trl import DPOTrainer, DPOConfig
import warnings
warnings.filterwarnings("ignore")


class DPOTrainer_SmolGraham:
    def __init__(self, sft_model_path: str = "SmolGraham", dpo_data_path: str = "dpo_pairs.csv"):
        """Initialize DPO trainer with SFT model and preference data."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sft_model_path = sft_model_path
        self.dpo_data_path = dpo_data_path
        
        print(f"Using device: {self.device}")
        print(f"Loading SFT model from: {sft_model_path}")
        print(f"Loading DPO data from: {dpo_data_path}")
        
        # Validate paths
        if not os.path.exists(sft_model_path):
            raise FileNotFoundError(f"SFT model directory not found: {sft_model_path}")
        if not os.path.exists(dpo_data_path):
            raise FileNotFoundError(f"DPO data file not found: {dpo_data_path}")
    
    def load_model_and_tokenizer(self):
        """Load the SFT model and tokenizer."""
        print("Loading model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.sft_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.sft_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Create reference model (copy of the SFT model for DPO)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.sft_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        print(f"Model loaded with {self.model.num_parameters():,} parameters")
        return self.model, self.ref_model, self.tokenizer
    
    def load_dpo_data(self):
        """Load and preprocess the DPO preference data."""
        print("Loading DPO preference data...")
        
        # Load the CSV
        df = pd.read_csv(self.dpo_data_path)
        print(f"Loaded {len(df)} preference pairs")
        
        # Filter out any error entries
        df = df[~df['chosen'].str.startswith('Error:', na=False)]
        df = df[~df['rejected'].str.startswith('Error:', na=False)]
        print(f"After filtering errors: {len(df)} preference pairs")
        
        # Filter by confidence (keep only high-confidence judgments)
        df = df[df['confidence'] >= 0]  # Keep judgments with confidence >= 6/10
        print(f"After confidence filtering: {len(df)} preference pairs")
        
        if len(df) == 0:
            raise ValueError("No valid preference pairs found after filtering!")
        
        # Convert to the format expected by DPOTrainer
        def format_conversation(prompt, response):
            """Format prompt and response as a conversation."""
            return [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
        
        # Prepare data for DPO training
        dpo_data = []
        for _, row in df.iterrows():
            dpo_data.append({
                "prompt": [{"role": "user", "content": row['prompt']}],
                "chosen": format_conversation(row['prompt'], row['chosen']),
                "rejected": format_conversation(row['prompt'], row['rejected']),
            })
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_list(dpo_data)
        
        # Split into train/validation
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        
        print(f"Training samples: {len(dataset['train'])}")
        print(f"Validation samples: {len(dataset['test'])}")
        
        return dataset
    
    def train_dpo(self, output_dir: str = "SmolGraham-DPO"):
        """Train the model using DPO."""
        print("Starting DPO training...")
        
        # Load model and data
        model, ref_model, tokenizer = self.load_model_and_tokenizer()
        dataset = self.load_dpo_data()
        
        # Configure DPO training
        training_args = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=1,  # Start with 1 epoch for DPO
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-7,  # Lower learning rate for DPO
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            eval_steps=100,
            save_steps=2000,
            save_total_limit=2,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["tensorboard"],
            fp16=False,
            dataloader_pin_memory=True,
            dataloader_num_workers=0,
            gradient_checkpointing=True,
            beta=0.2,  # KL penalty coefficient
            loss_type="sigmoid",  # DPO loss type
            remove_unused_columns=False,
            save_safetensors=True
        )
        
        # Initialize DPO trainer
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
        )
        
        # Clear cache before training
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Train the model
        print("🚀 Starting DPO training...")
        dpo_trainer.train()
        
        # Save the final model
        print(f"💾 Saving DPO model to {output_dir}")
        dpo_trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        return output_dir
    
    def test_model(self, model_dir: str):
        """Test the DPO-trained model with a sample generation."""
        print(f"🧪 Testing DPO model from {model_dir}")
        
        # Load the tokenizer from the trained DPO model to ensure consistency
        dpo_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load the trained model for inference
        pipe = pipeline(
            "text-generation",
            model=model_dir,
            device=0 if self.device == "cuda" else "cpu",
            torch_dtype=torch.bfloat16
        )
        
        # Test prompts
        test_prompts = [
            "Write a Paul Graham essay about the power of AI",
            "Write a Paul Graham essay about startups and entrepreneurship",
            "Write a Paul Graham essay about the importance of simplicity in design"
        ]
        
        print("\n" + "="*50)
        print("🎯 TESTING DPO-TRAINED MODEL")
        print("="*50)
        
        for i, test_prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i} ---")
            print(f"Prompt: {test_prompt}")
            print("\nGenerated Essay:")
            print("-" * 30)
            
            # Format as conversation
            messages = [{"role": "user", "content": test_prompt}]
            formatted_prompt = dpo_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Generate
            output = pipe(
                formatted_prompt,
                max_new_tokens=1024,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )[0]["generated_text"]
            
            # Extract only the generated part
            generated_text = output[len(formatted_prompt):].strip()
            print(generated_text)
            print("-" * 30)
    
    def run_full_pipeline(self, output_dir: str = "SmolGraham-DPO"):
        """Run the complete DPO training pipeline."""
        print("🔥 Starting DPO Training Pipeline")
        print("=" * 50)
        
        try:
            # Train with DPO
            final_model_dir = self.train_dpo(output_dir)
            
            # Test the model
            self.test_model(final_model_dir)
            
            print(f"\n✅ DPO training complete! Model saved to: {final_model_dir}")
            print("\nYou can now use this model for improved Paul Graham essay generation!")
            
            return final_model_dir
            
        except Exception as e:
            print(f"❌ Error during DPO training: {e}")
            raise


def main():
    """Main function to run DPO training."""
    print("🎯 SmolGraham DPO Training")
    print("=" * 30)
    
    # Check if SFT model exists
    sft_model_path = "SmolGraham"
    if not os.path.exists(sft_model_path):
        print(f"❌ SFT model not found at {sft_model_path}")
        print("Please run sft.py first to create the base model.")
        return
    
    # Check if DPO data exists
    dpo_data_path = "dpo_pairs.csv"
    if not os.path.exists(dpo_data_path):
        print(f"❌ DPO data not found at {dpo_data_path}")
        print("Please run generate_dpo_data.py first to create preference pairs.")
        return
    
    print("✅ All required files found!")
    print(f"✅ SFT model: {sft_model_path}")
    print(f"✅ DPO data: {dpo_data_path}")
    
    # Initialize and run DPO training
    trainer = DPOTrainer_SmolGraham(sft_model_path, dpo_data_path)
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main() 