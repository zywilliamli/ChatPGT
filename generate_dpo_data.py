"""
DPO Data Generator with Hybrid Model Support

This script generates synthetic Paul Graham essays using a hybrid approach:
- OpenAI models: Run in parallel for fast API-based generation
- Local models: Run sequentially to avoid GPU memory conflicts

Key features:
- Automatic CUDA detection and local model loading
- GPU-safe sequential processing for local models  
- Parallel processing for OpenAI API calls
- Rate limiting and error handling
- Memory management for large models

Usage:
    python generate_dpo_data.py  # Automatic hybrid mode
    
For debugging or very limited GPU memory:
    generator = DPODataGenerator(force_sequential=True)
"""

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
    def __init__(self, openai_api_key: str = None, max_workers: int = 10, force_sequential: bool = False):
        """Initialize the DPO data generator with OpenAI client and optional local models."""
        self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.max_workers = max_workers  # Number of parallel workers for OpenAI
        self.force_sequential = force_sequential  # Force all processing to be sequential
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
                "Qwen/Qwen3-8B",
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
        
        # Clear GPU cache before loading
        try:
            torch.cuda.empty_cache()
        except:
            pass
        
        successfully_loaded = []
        
        for model_name in self.local_model_names:
            try:
                print(f"  Loading {model_name}...")
                
                # Check available GPU memory
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    print(f"    Available GPU memory: {gpu_memory / 1e9:.1f}GB")
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model with memory management
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    # Add memory management options
                    attn_implementation="flash_attention_2" if "llama" in model_name.lower() else None
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
                successfully_loaded.append(model_name)
                print(f"  ✅ {model_name} loaded successfully")
                
                # Show GPU memory usage after loading
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated(0)
                    print(f"    GPU memory used: {memory_used / 1e9:.1f}GB")
                
            except Exception as e:
                print(f"  ❌ Failed to load {model_name}: {e}")
                print(f"    This might be due to insufficient GPU memory or model availability")
                # Clear cache and continue with next model
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
        
        # Update the list to only include successfully loaded models
        self.local_model_names = successfully_loaded
        
        print(f"Successfully loaded {len(self.local_models)} local models")
        if len(self.local_models) == 0 and self.device == "cuda":
            print("⚠️  No local models loaded. Will use OpenAI models only.")
    
    def _is_local_model(self, model_name: str) -> bool:
        """Check if a model is a local Hugging Face model."""
        return model_name in self.local_model_names
    
    def _clean_thinking_tags(self, text: str) -> str:
        """Remove thinking tags from generated text (for models like Qwen that use <think>...</think>)."""
        import re
        
        original_length = len(text)
        
        # Remove properly closed thinking tags and their content
        # This handles both single-line and multi-line thinking sections
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Handle incomplete thinking tags - be more conservative
        # Remove from <think> to the end of the current paragraph (double newline) or end of text
        # This prevents removing valid content that might follow
        cleaned_text = re.sub(r'<think[^>]*>.*?(?:\n\s*\n|$)', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up any extra whitespace that might be left
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)  # Multiple newlines to double
        cleaned_text = cleaned_text.strip()
        
        # Log if thinking tags were found and removed
        if len(cleaned_text) < original_length:
            removed_chars = original_length - len(cleaned_text)
            print(f"    🧠 Cleaned thinking tags (removed {removed_chars} characters)")
        
        return cleaned_text
    
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
            
            # Generate response with GPU memory management
            with torch.cuda.device(0) if torch.cuda.is_available() else torch.no_grad():
                outputs = pipe(
                    formatted_prompt,
                    max_new_tokens=1500,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    # Add batch size limitation for memory management
                    batch_size=1
                )
            
            # Extract the generated text
            generated_text = outputs[0]["generated_text"]
            
            # Remove the prompt from the generated text
            if generated_text.startswith(formatted_prompt):
                generated_text = generated_text[len(formatted_prompt):].strip()
            
            # Clean up thinking tags (especially for Qwen models)
            generated_text = self._clean_thinking_tags(generated_text)
            
            return generated_text
            
        except Exception as e:
            print(f"Error generating essay with local model '{model}' for topic '{topic}': {e}")
            # Clear GPU cache on error
            try:
                torch.cuda.empty_cache()
            except:
                pass
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
        """Generate essays for all topics using hybrid approach: parallel OpenAI, sequential local."""
        print(f"Generating essays for {len(topics)} topics using {len(self.essay_models)} models...")
        print(f"  OpenAI models: {self.openai_models}")
        print(f"  Local models: {self.local_model_names}")
        
        all_results = []
        
        # Step 1: Generate OpenAI essays in parallel
        if self.openai_models:
            print(f"\n🌐 Generating OpenAI essays in parallel ({self.max_workers} workers)...")
            openai_results = self._generate_openai_essays_parallel(topics)
            all_results.extend(openai_results)
        
        # Step 2: Generate local model essays sequentially
        if self.local_model_names and self.local_models:
            print(f"\n🖥️  Generating local model essays sequentially (GPU-safe)...")
            local_results = self._generate_local_essays_sequential(topics)
            all_results.extend(local_results)
        
        # Save to CSV
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Saved {len(all_results)} essays to {output_csv}")
        return output_csv
    
    def _generate_openai_essays_parallel(self, topics: List[str]) -> List[Dict]:
        """Generate essays using OpenAI models in parallel (or sequential if forced)."""
        # Create OpenAI tasks
        openai_tasks = []
        for topic in topics:
            for model in self.openai_models:
                openai_tasks.append((topic, model))
        
        results = []
        
        if self.force_sequential:
            # Sequential processing for all models (useful for debugging or very limited resources)
            with tqdm(total=len(openai_tasks), desc="OpenAI essays (sequential)") as pbar:
                for topic, model in openai_tasks:
                    try:
                        result = self._generate_essay_task(topic, model)
                        results.append(result)
                    except Exception as e:
                        print(f"Error with OpenAI task {topic}/{model}: {e}")
                        results.append({
                            'topic': topic,
                            'model': model,
                            'essay': f"Error: {str(e)}",
                            'prompt_template': f"Error generating for {topic}"
                        })
                    finally:
                        pbar.update(1)
        else:
            # Normal parallel processing for OpenAI
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all OpenAI tasks
                future_to_task = {
                    executor.submit(self._generate_essay_task, topic, model): (topic, model)
                    for topic, model in openai_tasks
                }
                
                # Process completed tasks with progress bar
                with tqdm(total=len(openai_tasks), desc="OpenAI essays (parallel)") as pbar:
                    for future in as_completed(future_to_task):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            topic, model = future_to_task[future]
                            print(f"Error with OpenAI task {topic}/{model}: {e}")
                            # Add error result to maintain count
                            results.append({
                                'topic': topic,
                                'model': model,
                                'essay': f"Error: {str(e)}",
                                'prompt_template': f"Error generating for {topic}"
                            })
                        finally:
                            pbar.update(1)
        
        return results
    
    def _generate_local_essays_sequential(self, topics: List[str]) -> List[Dict]:
        """Generate essays using local models sequentially (GPU-safe)."""
        results = []
        total_local_tasks = len(topics) * len(self.local_model_names)
        
        with tqdm(total=total_local_tasks, desc="Local model essays") as pbar:
            for topic in topics:
                for model in self.local_model_names:
                    try:
                        result = self._generate_essay_task(topic, model)
                        results.append(result)
                    except Exception as e:
                        print(f"Error with local task {topic}/{model}: {e}")
                        # Add error result to maintain count
                        results.append({
                            'topic': topic,
                            'model': model,
                            'essay': f"Error: {str(e)}",
                            'prompt_template': f"Error generating for {topic}"
                        })
                    finally:
                        pbar.update(1)
                        
                        # Optional: Clear GPU cache after each generation to prevent memory buildup
                        if hasattr(self, 'device') and self.device == "cuda":
                            try:
                                import torch
                                torch.cuda.empty_cache()
                            except:
                                pass
        
        return results
    
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
    
    # Initialize generator with hybrid processing
    # OpenAI models run in parallel, local models run sequentially (GPU-safe)
    generator = DPODataGenerator(max_workers=10)
    
    print(f"\n📊 Model Configuration:")
    print(f"  Total models: {len(generator.essay_models)}")
    print(f"  OpenAI models: {len(generator.openai_models)}")
    print(f"  Local models: {len(generator.local_model_names)}")
    
    print(f"\n🔧 Processing Strategy:")
    if generator.device == "cuda" and generator.local_models:
        print(f"  🔥 CUDA detected! Using {len(generator.local_models)} local models")
        print(f"  🌐 OpenAI models: Parallel processing ({generator.max_workers} workers)")
        print(f"  🖥️  Local models: Sequential processing (GPU-safe)")
    elif generator.device == "cuda" and not generator.local_models:
        print(f"  ⚠️  CUDA detected but no local models loaded")
        print(f"  💡 Install transformers/torch for local model support")
        print(f"  🌐 OpenAI models: Parallel processing ({generator.max_workers} workers)")
    else:
        print(f"  💻 CPU mode - using OpenAI models only")
        print(f"  🌐 OpenAI models: Parallel processing ({generator.max_workers} workers)")
    
    # Run the full pipeline
    generator.generate_full_pipeline(num_topics=10)


if __name__ == "__main__":
    main() 