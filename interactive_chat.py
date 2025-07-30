#!/usr/bin/env python3
"""
Interactive chat script for testing SFT and DPO trained models.
Supports loading models from either sft.py or dpo.py outputs.
"""
import os
import sys
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
import warnings
warnings.filterwarnings("ignore")


class InteractiveChat:
    def __init__(self, model_path: str, model_type: str = "auto"):
        """
        Initialize interactive chat interface.
        
        Args:
            model_path: Path to the trained model directory
            model_type: Type of model ("sft", "dpo", or "auto" for auto-detection)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🤖 Interactive Chat Interface")
        print(f"📁 Model path: {model_path}")
        print(f"🔧 Model type: {model_type}")
        print(f"💻 Device: {self.device}")
        print("-" * 50)
        
        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Auto-detect model type if needed
        if model_type == "auto":
            self.model_type = self._detect_model_type()
            print(f"🔍 Auto-detected model type: {self.model_type}")
        
        # Load model and tokenizer
        self.load_model()
        
        # Setup pipeline
        self.setup_pipeline()
        
        print("✅ Model loaded successfully!")
        print("💬 Type your messages below (type 'quit', 'exit', or Ctrl+C to exit)")
        print("-" * 50)
    
    def _detect_model_type(self) -> str:
        """Auto-detect whether this is an SFT or DPO model."""
        # Check for DPO-specific files or indicators
        config_file = os.path.join(self.model_path, "config.json")
        if os.path.exists(config_file):
            try:
                import json
                with open(config_file, 'r') as f:
                    config = json.load(f)
                # Look for DPO-specific indicators in config
                if "dpo" in str(config).lower():
                    return "dpo"
            except:
                pass
        
        # Check for DPO checkpoint files
        checkpoint_dirs = [d for d in os.listdir(self.model_path) if d.startswith("checkpoint")]
        if checkpoint_dirs:
            # DPO training typically creates more checkpoints
            return "dpo" if len(checkpoint_dirs) > 5 else "sft"
        
        # Default to SFT
        return "sft"
    
    def load_model(self):
        """Load the model and tokenizer."""
        print("📥 Loading model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Keep original chat template to maintain model behavior consistency
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
    
    def setup_pipeline(self):
        """Setup the text generation pipeline."""
        print("⚙️  Setting up generation pipeline...")
        
        # Create pipeline without generation parameters (set them during generation instead)
        pipeline_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
        }
        
        # Only specify device for CPU usage (device_map="auto" handles CUDA automatically)
        if self.device == "cpu":
            pipeline_kwargs["device"] = -1
        
        self.pipe = pipeline("text-generation", **pipeline_kwargs)
    
    def format_prompt(self, user_input: str) -> str:
        """Format user input into the appropriate chat format."""
        messages = [
            {"role": "user", "content": user_input}
        ]
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except Exception as e:
            print(f"⚠️  Warning: Could not apply chat template ({e}), using simple format")
            return f"### User: {user_input}\\n### Assistant: "
    
    def generate_response(self, user_input: str) -> str:
        """Generate a response to user input."""
        prompt = self.format_prompt(user_input)
        
        try:
            # Generate response
            outputs = self.pipe(
                prompt,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            
            # Extract the generated text
            generated_text = outputs[0]["generated_text"]
            
            # Remove the prompt from the response
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()
            
            return response
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def chat_loop(self):
        """Main interactive chat loop."""
        conversation_history = []
        
        while True:
            try:
                # Get user input
                user_input = input("\\n👤 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Add to conversation history
                conversation_history.append({"role": "user", "content": user_input})
                
                # Generate response
                print("🤖 Assistant: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                
                # Add response to history
                conversation_history.append({"role": "assistant", "content": response})
                
                # Keep conversation history reasonable
                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]
                    
            except KeyboardInterrupt:
                print("\\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\\n❌ Error: {str(e)}")
                continue
    
    def run(self):
        """Run the interactive chat interface."""
        self.chat_loop()


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with SFT/DPO trained models")
    parser.add_argument("model_path", help="Path to the trained model directory")
    parser.add_argument(
        "--type", 
        choices=["sft", "dpo", "auto"], 
        default="auto",
        help="Type of model (auto-detect by default)"
    )
    parser.add_argument(
        "--examples", 
        action="store_true", 
        help="Show example prompts for Paul Graham style essays"
    )
    
    args = parser.parse_args()
    
    if args.examples:
        print("📝 Example prompts for Paul Graham style essays:")
        print("   • Write a Paul Graham essay about the future of AI")
        print("   • What would Paul Graham say about remote work?")
        print("   • Write an essay about startup ideas in the style of Paul Graham")
        print("   • How would Paul Graham analyze the current tech bubble?")
        print("   • Write about programming languages like Paul Graham would")
        print()
    
    try:
        # Create and run interactive chat
        chat = InteractiveChat(args.model_path, args.type)
        chat.run()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you've trained a model first using sft.py or dpo.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()