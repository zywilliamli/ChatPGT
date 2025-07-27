import os
import random
import pandas as pd
import numpy as np
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import time
import csv
from typing import List, Dict, Tuple, Union
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Optional imports for local models
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers/torch not available. Local models will be disabled.")


class DPODataGenerator:
    def __init__(self, openai_api_key: str = None, max_workers: int = 10):
        """Initialize the DPO data generator with OpenAI client and optional local models."""
        self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.max_workers = max_workers  # Number of parallel workers
        self.rate_limit_lock = threading.Lock()
        self.last_request_time = 0
        self.min_request_interval = 0.05  # 50ms between requests for rate limiting
        
        # Check for CUDA and load local models if available
        self.device = "cuda" if HF_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.local_models = {}
        self.local_tokenizers = {}
        
        print(f"Using device: {self.device}")
        
        # OpenAI models for essay generation
        self.openai_models = [
            "gpt-3.5-turbo-0125",
            "gpt-4.1-2025-04-14",
            "gpt-4.1-nano-2025-04-14",
        ]
        
        # Local models to load if CUDA is available
        self.local_model_names = []
        if self.device == "cuda":
            self.local_model_names = [
                "HuggingFaceTB/SmolLM3-3B",
                "meta-llama/Meta-Llama-3-8B-Instruct",
            ]
            self._load_local_models()
        
        # Combined list of all models
        self.essay_models = self.openai_models + self.local_model_names
        
        # Model for judging preferences (always OpenAI)
        self.judge_model = "o4-mini-2025-04-16"
        
        # Essay prompt templates
        self.prompt_templates = [
            "Write a Paul Graham essay about {topic}",
            "Generate an essay in the style of Paul Graham about {topic}",
            "Compose a Paul Graham-style essay on {topic}",
            "Create an essay about {topic} in the writing style of Paul Graham",
            "Write an essay on {topic} as if you were Paul Graham",
        ]
    
    def _load_local_models(self):
        """Load local Hugging Face models if CUDA is available."""
        if not HF_AVAILABLE or self.device != "cuda":
            return
        
        print("Loading local models...")
        for model_name in self.local_model_names:
            try:
                print(f"  Loading {model_name}...")
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # Create pipeline
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device_map="auto",
                    torch_dtype=torch.bfloat16
                )
                
                self.local_models[model_name] = pipe
                self.local_tokenizers[model_name] = tokenizer
                print(f"  ✓ {model_name} loaded successfully")
                
            except Exception as e:
                print(f"  ❌ Failed to load {model_name}: {e}")
                # Remove from list if failed to load
                if model_name in self.local_model_names:
                    self.local_model_names.remove(model_name)
        
        print(f"Successfully loaded {len(self.local_models)} local models")
    
    def _is_local_model(self, model_name: str) -> bool:
        """Check if a model is a local Hugging Face model."""
        return model_name in self.local_model_names
    
    def load_wikipedia_data(self, num_samples: int = 10) -> List[str]:
        """Load Wikipedia dataset and extract random topics."""
        print("Loading Wikipedia dataset...")
        
        # Load the specific subset
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        dataset = dataset.shuffle(seed=42, buffer_size=1000)
        dataset = dataset.take(num_samples*10)
        dataset = dataset.to_list()
        
        print(f"Dataset loaded with {len(dataset)} articles")
        
        # Extract titles as topics, filtering out very short or very long ones
        topics = []
        for item in dataset:
            title = item.get('title', '').strip()
            if title and 5 <= len(title) <= 100:  # Reasonable title length
                topics.append(title)
        
        print(f"Found {len(topics)} valid topics")
        
        # Randomly sample the requested number
        if len(topics) > num_samples:
            topics = random.sample(topics, num_samples)
        
        print(f"Selected {len(topics)} topics for essay generation")
        return topics
    
    def _rate_limited_request(self, request_func, *args, **kwargs):
        """Execute a request with rate limiting (only for OpenAI requests)."""
        with self.rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            self.last_request_time = time.time()
        
        return request_func(*args, **kwargs)
    
    def _generate_essay_local(self, topic: str, model: str) -> str:
        """Generate an essay using a local Hugging Face model."""
        if model not in self.local_models:
            return f"Error: Local model {model} not loaded"
        
        try:
            # Randomly select a prompt template
            template = random.choice(self.prompt_templates)
            prompt = template.format(topic=topic)
            
            # Create messages for chat template
            messages = [
                {"role": "system", "content": "You are an expert writer who can write in the style of Paul Graham, the famous essayist and Y Combinator founder. Write thoughtful, insightful essays that capture his voice, style, and perspective."},
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template if tokenizer supports it
            tokenizer = self.local_tokenizers[model]
            pipe = self.local_models[model]
            
            if hasattr(tokenizer, 'apply_chat_template'):
                try:
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except:
                    # Fallback if chat template fails
                    formatted_prompt = f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"
            else:
                formatted_prompt = f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"
            
            # Generate response
            outputs = pipe(
                formatted_prompt,
                max_new_tokens=1500,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Extract the generated text
            generated_text = outputs[0]["generated_text"]
            
            # Remove the prompt from the generated text
            if generated_text.startswith(formatted_prompt):
                generated_text = generated_text[len(formatted_prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            print(f"Error generating essay with local model '{model}' for topic '{topic}': {e}")
            return f"Error: {str(e)}"
    
    def generate_essay(self, topic: str, model: str) -> str:
        """Generate a single essay for a given topic using specified model (OpenAI or local)."""
        # Route to appropriate generation method
        if self._is_local_model(model):
            return self._generate_essay_local(topic, model)
        else:
            return self._generate_essay_openai(topic, model)
    
    def _generate_essay_openai(self, topic: str, model: str) -> str:
        """Generate an essay using an OpenAI model."""
        # Randomly select a prompt template
        template = random.choice(self.prompt_templates)
        prompt = template.format(topic=topic)
        
        try:
            response = self._rate_limited_request(
                self.client.chat.completions.create,
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert writer who can write in the style of Paul Graham, the famous essayist and Y Combinator founder. Write thoughtful, insightful essays that capture his voice, style, and perspective."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.8,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating essay for topic '{topic}' with OpenAI model '{model}': {e}")
            return f"Error: {str(e)}"
    
    def _generate_essay_task(self, topic: str, model: str) -> Dict:
        """Wrapper for generate_essay that returns a result dict for parallel processing."""
        essay = self.generate_essay(topic, model)
        return {
            'topic': topic,
            'model': model,
            'essay': essay,
            'prompt_template': random.choice(self.prompt_templates).format(topic=topic)
        }
    
    def generate_all_essays(self, topics: List[str], output_csv: str = "paul_graham_essays.csv"):
        """Generate essays for all topics using all models in parallel."""
        print(f"Generating essays for {len(topics)} topics using {len(self.essay_models)} models...")
        print(f"  OpenAI models: {self.openai_models}")
        print(f"  Local models: {self.local_model_names}")
        print(f"Using {self.max_workers} parallel workers")
        
        # Create all tasks
        tasks = []
        for topic in topics:
            for model in self.essay_models:
                tasks.append((topic, model))
        
        results = []
        total_requests = len(tasks)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._generate_essay_task, topic, model): (topic, model)
                for topic, model in tasks
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=total_requests, desc="Generating essays") as pbar:
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        topic, model = future_to_task[future]
                        print(f"Error with task {topic}/{model}: {e}")
                        # Add error result to maintain count
                        results.append({
                            'topic': topic,
                            'model': model,
                            'essay': f"Error: {str(e)}",
                            'prompt_template': f"Error generating for {topic}"
                        })
                    finally:
                        pbar.update(1)
        
        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(results)} essays to {output_csv}")
        return output_csv
    
    def judge_essay_pair(self, topic: str, essay1: str, essay2: str, model1: str, model2: str) -> Dict:
        """Use GPT to judge which essay is better."""
        judgment_prompt = f"""
You are evaluating two essays written in the style of Paul Graham about the topic: "{topic}"

Please evaluate these essays based on:
1. How well they capture Paul Graham's writing style and voice
2. Clarity and depth of insights
3. Overall quality and coherence
4. Authenticity to Paul Graham's perspective

Essay A (from {model1}):
{essay1}

Essay B (from {model2}):
{essay2}

Respond with a JSON object containing:
- "winner": "A" or "B" (which essay is better overall)
- "reasoning": A brief explanation of why you chose that essay
- "confidence": A number from 1-10 indicating how confident you are in this judgment
"""
        
        try:
            response = self._rate_limited_request(
                self.client.chat.completions.create,
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert judge of writing quality and style. Provide fair, detailed evaluations."},
                    {"role": "user", "content": judgment_prompt}
                ],
                max_completion_tokens=500,
            )
            
            # Parse the JSON response
            judgment_text = response.choices[0].message.content.strip()
            # Extract JSON from the response (in case there's extra text)
            start_idx = judgment_text.find('{')
            end_idx = judgment_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = judgment_text[start_idx:end_idx]
                judgment = json.loads(json_str)
            else:
                # Fallback if JSON parsing fails
                judgment = {
                    "winner": "A" if "Essay A" in judgment_text else "B",
                    "reasoning": judgment_text,
                    "confidence": 5
                }
                
            return judgment
        except Exception as e:
            print(f"Error judging essay pair: {e}")
            return {
                "winner": "A",  # Default fallback
                "reasoning": f"Error in judgment: {str(e)}",
                "confidence": 1
            }
    
    def _judge_essay_pair_task(self, topic: str, essay1: str, essay2: str, model1: str, model2: str) -> Dict:
        """Wrapper for judge_essay_pair that returns a complete DPO pair dict."""
        judgment = self.judge_essay_pair(topic, essay1, essay2, model1, model2)
        
        # Create DPO pair
        if judgment["winner"] == "A":
            chosen = essay1
            rejected = essay2
            chosen_model = model1
            rejected_model = model2
        else:
            chosen = essay2
            rejected = essay1
            chosen_model = model2
            rejected_model = model1
        
        return {
            'topic': topic,
            'prompt': f"Write a Paul Graham essay about {topic}",
            'chosen': chosen,
            'rejected': rejected,
            'chosen_model': chosen_model,
            'rejected_model': rejected_model,
            'judgment_reasoning': judgment["reasoning"],
            'confidence': judgment["confidence"]
        }
    
    def create_dpo_pairs(self, essays_csv: str, output_csv: str = "dpo_pairs.csv"):
        """Create DPO preference pairs from the generated essays in parallel."""
        print(f"Creating DPO pairs from {essays_csv}...")
        print(f"Using {self.max_workers} parallel workers for judgments")
        
        # Load the essays
        df = pd.read_csv(essays_csv)
        
        # Create all comparison tasks
        comparison_tasks = []
        topics = df['topic'].unique()
        
        print("Preparing comparison tasks...")
        for topic in topics:
            topic_essays = df[df['topic'] == topic]
            
            # Create all possible pairs for this topic
            models = topic_essays['model'].tolist()
            essays = topic_essays['essay'].tolist()
            
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    model1, essay1 = models[i], essays[i]
                    model2, essay2 = models[j], essays[j]
                    
                    # Skip if either essay has an error
                    if essay1.startswith("Error:") or essay2.startswith("Error:"):
                        continue
                    
                    comparison_tasks.append((topic, essay1, essay2, model1, model2))
        
        print(f"Created {len(comparison_tasks)} comparison tasks")
        
        # Process comparisons in parallel
        dpo_pairs = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all judgment tasks
            future_to_task = {
                executor.submit(self._judge_essay_pair_task, topic, essay1, essay2, model1, model2): (topic, model1, model2)
                for topic, essay1, essay2, model1, model2 in comparison_tasks
            }
            
            # Process completed judgments with progress bar
            with tqdm(total=len(comparison_tasks), desc="Creating DPO pairs") as pbar:
                for future in as_completed(future_to_task):
                    try:
                        dpo_pair = future.result()
                        dpo_pairs.append(dpo_pair)
                    except Exception as e:
                        topic, model1, model2 = future_to_task[future]
                        print(f"Error judging pair {topic}/{model1}/{model2}: {e}")
                        # Add error pair to maintain count
                        dpo_pairs.append({
                            'topic': topic,
                            'prompt': f"Write a Paul Graham essay about {topic}",
                            'chosen': f"Error in judgment: {str(e)}",
                            'rejected': f"Error in judgment: {str(e)}",
                            'chosen_model': model1,
                            'rejected_model': model2,
                            'judgment_reasoning': f"Error: {str(e)}",
                            'confidence': 1
                        })
                    finally:
                        pbar.update(1)
        
        # Save DPO pairs
        dpo_df = pd.DataFrame(dpo_pairs)
        dpo_df.to_csv(output_csv, index=False)
        print(f"Created {len(dpo_pairs)} DPO pairs and saved to {output_csv}")
        return output_csv
    
    def generate_full_pipeline(self, num_topics: int = 1000):
        """Run the complete DPO data generation pipeline."""
        print("Starting DPO data generation pipeline...")
        
        # Step 1: Load topics
        topics = self.load_wikipedia_data(num_topics)
        
        # Step 2: Generate essays
        essays_csv = self.generate_all_essays(topics)
        
        # Step 3: Create DPO pairs
        dpo_csv = self.create_dpo_pairs(essays_csv)
        
        print(f"Pipeline complete! DPO data saved to {dpo_csv}")
        return dpo_csv


def main():
    """Main function to run the DPO data generation."""
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set the OPENAI_API_KEY environment variable")
        return
    
    print("🚀 Initializing DPO Data Generator with Hybrid Models")
    print("=" * 60)
    
    # Initialize generator with parallel processing
    # Adjust max_workers based on your rate limits (default 10 should be safe)
    generator = DPODataGenerator(max_workers=10)
    
    print(f"\n📊 Model Configuration:")
    print(f"  Total models: {len(generator.essay_models)}")
    print(f"  OpenAI models: {len(generator.openai_models)}")
    print(f"  Local models: {len(generator.local_model_names)}")
    
    if generator.device == "cuda" and generator.local_models:
        print(f"\n🔥 CUDA detected! Using {len(generator.local_models)} local models for faster generation")
    elif generator.device == "cuda" and not generator.local_models:
        print(f"\n⚠️  CUDA detected but no local models loaded (check transformers installation)")
    else:
        print(f"\n💻 Using CPU mode - only OpenAI models available")
    
    # Run the full pipeline
    generator.generate_full_pipeline(num_topics=10)


if __name__ == "__main__":
    main() 