#!/usr/bin/env python3
"""
Interactive chat script for testing SFT and DPO trained models.
Supports loading models from either local directories or Hugging Face Hub.
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
            model_path: Path to the trained model directory OR Hugging Face Hub model name
            model_type: Type of model ("sft", "dpo", or "auto" for auto-detection)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Detect if this is a Hub model or local path
        self.is_hub_model = self._is_hub_model(model_path)
        
        print(f"🤖 Interactive Chat Interface")
        print(f"📁 Model: {model_path}")
        print(f"🌐 Source: {'Hugging Face Hub' if self.is_hub_model else 'Local directory'}")
        print(f"🔧 Model type: {model_type}")
        print(f"💻 Device: {self.device}")
        print("-" * 50)
        
        # Validate model path/name
        if not self.is_hub_model and not os.path.exists(model_path):
            raise FileNotFoundError(f"Local model directory not found: {model_path}")
        
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
    
    def _is_hub_model(self, model_path: str) -> bool:
        """Detect if model_path is a Hub model name or local path."""
        # Hub model names typically contain a slash (username/model-name)
        # and don't exist as local directories
        if "/" in model_path and not os.path.exists(model_path):
            return True
        # Also check for common hub patterns
        if model_path.startswith(("hf://", "huggingface://")) or \
           (len(model_path.split("/")) == 2 and not os.path.exists(model_path)):
            return True
        return False
    
    def _detect_model_type(self) -> str:
        """Auto-detect whether this is an SFT or DPO model."""
        if self.is_hub_model:
            # For hub models, use naming convention hints
            if "dpo" in self.model_path.lower():
                return "dpo"
            elif "sft" in self.model_path.lower():
                return "sft"
            else:
                return "sft"  # Default for hub models
        
        # For local models, use existing detection logic
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
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
        except Exception as e:
            if self.is_hub_model:
                print(f"❌ Failed to load model from Hugging Face Hub: {e}")
                print("💡 Make sure the model exists and you have access to it")
                print("💡 For private models, login with `huggingface-cli login`")
            else:
                print(f"❌ Failed to load local model: {e}")
            raise
    
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
            return f"### User: {user_input}\n### Assistant: "
    
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
            
            return generated_text
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def chat_loop(self):
        """Main interactive chat loop."""
        conversation_history = []
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
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
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                continue
    
    def run(self):
        """Run the interactive chat interface."""
        self.chat_loop()


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with SFT/DPO trained models")
    parser.add_argument(
        "model_path", 
        help="Path to the trained model directory OR Hugging Face Hub model name (e.g., 'username/model-name')"
    )
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
        print("📋 Example model names:")
        print("   • Local: SmolGraham or SmolGraham-DPO")
        print("   • Hub: username/SmolGraham-SFT or username/SmolGraham-DPO")
        print()
    
    try:
        # Create and run interactive chat
        chat = InteractiveChat(args.model_path, args.type)
        chat.run()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you've trained a model first using sft.py or dpo.py")
        print("💡 Or provide a valid Hugging Face Hub model name")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()