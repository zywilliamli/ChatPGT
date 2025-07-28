import os
import random
import pandas as pd
import numpy as np
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import time
import csv
from typing import List, Dict, Tuple
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import httpx


class DPODataGenerator:
    def __init__(self, openai_api_key: str = None, openrouter_api_key: str = None, max_workers: int = 50):
        """Initialize the DPO data generator with OpenAI and OpenRouter clients."""
        # OpenAI client
        self.openai_client = None
        if openai_api_key or os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        
        # OpenRouter client (uses OpenAI-compatible API)
        self.openrouter_client = None
        if openrouter_api_key or os.getenv("OPENROUTER_API_KEY"):
            self.openrouter_client = OpenAI(
                api_key=openrouter_api_key or os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )
        
        self.max_workers = max_workers  # Number of parallel workers
        self.rate_limit_lock = threading.Lock()
        self.last_request_time = 0
        self.min_request_interval = 0.05  # 50ms between requests for rate limiting
        
        # OpenAI models
        self.openai_models = [
            "gpt-3.5-turbo-0125",
            "gpt-4.1-2025-04-14", 
            "gpt-4.1-nano-2025-04-14",
        ]
        
        # OpenRouter models (open source)
        self.openrouter_models = [
            "qwen/qwen3-8b",
            "mistralai/mistral-small-3.2-24b-instruct",
        ]
        
        # Combined models list for essay generation
        self.essay_models = []
        if self.openai_client:
            self.essay_models.extend(self.openai_models)
        if self.openrouter_client:
            self.essay_models.extend(self.openrouter_models)
        
        # Model for judging preferences (using OpenAI for consistency)
        self.judge_model = "o4-mini-2025-04-16"
        
        # Essay prompt templates
        self.prompt_templates = [
            "Write a Paul Graham essay about {topic}",
            "Generate an essay in the style of Paul Graham about {topic}",
            "Compose a Paul Graham-style essay on {topic}",
            "Create an essay about {topic} in the writing style of Paul Graham",
            "Write an essay on {topic} as if you were Paul Graham",
        ]
    
    def load_wikipedia_data(self, num_samples: int = 1000) -> List[str]:
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
        """Execute a request with rate limiting."""
        with self.rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            self.last_request_time = time.time()
        
        return request_func(*args, **kwargs)
    
    def _get_client_for_model(self, model: str):
        """Get the appropriate client for the given model."""
        if model in self.openai_models:
            if self.openai_client is None:
                raise ValueError(f"OpenAI client not configured, but model {model} requires it. Please set OPENAI_API_KEY.")
            return self.openai_client
        elif model in self.openrouter_models:
            if self.openrouter_client is None:
                raise ValueError(f"OpenRouter client not configured, but model {model} requires it. Please set OPENROUTER_API_KEY.")
            return self.openrouter_client
        else:
            # Default to OpenAI for unknown models
            if self.openai_client is None:
                raise ValueError(f"No API clients configured. Please set OPENAI_API_KEY and/or OPENROUTER_API_KEY.")
            return self.openai_client
    
    def generate_essay(self, topic: str, model: str) -> str:
        """Generate a single essay for a given topic using specified model."""
        # Randomly select a prompt template
        template = random.choice(self.prompt_templates)
        prompt = template.format(topic=topic)
        
        try:
            # Get the appropriate client for this model
            client = self._get_client_for_model(model)
            
            response = self._rate_limited_request(
                client.chat.completions.create,
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
            print(f"Error generating essay for topic '{topic}' with model '{model}': {e}")
            return f"Error: {str(e)}"
    
    def _generate_essay_task(self, topic: str, model: str) -> Dict:
        """Wrapper for generate_essay that returns a result dict for parallel processing."""
        essay = self.generate_essay(topic, model)
        
        # Ensure essay is a valid string
        if essay is None or (isinstance(essay, float) and pd.isna(essay)):
            essay = "Error: Failed to generate essay"
        elif not isinstance(essay, str):
            essay = str(essay) if essay is not None else "Error: No essay returned"
        
        return {
            'topic': topic,
            'model': model,
            'essay': essay,
            'prompt_template': random.choice(self.prompt_templates).format(topic=topic)
        }
    
    def generate_all_essays(self, topics: List[str], output_csv: str = "paul_graham_essays.csv"):
        """Generate essays for all topics using all models in parallel."""
        print(f"Generating essays for {len(topics)} topics using {len(self.essay_models)} models...")
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
            if self.openai_client is None:
                raise ValueError("OpenAI client not configured. Judging requires OPENAI_API_KEY to be set.")
            
            response = self._rate_limited_request(
                self.openai_client.chat.completions.create,
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
        
        # Clean the data - replace NaN values with error strings
        df['essay'] = df['essay'].fillna("Error: Missing essay data")
        df['topic'] = df['topic'].fillna("Unknown topic")
        df['model'] = df['model'].fillna("Unknown model")
        
        print(f"Loaded {len(df)} essays from CSV")
        
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
                    
                    # Convert to string and handle NaN/None values
                    essay1_str = str(essay1) if essay1 is not None and pd.notna(essay1) else "Error: Missing essay"
                    essay2_str = str(essay2) if essay2 is not None and pd.notna(essay2) else "Error: Missing essay"
                    
                    # Skip if either essay has an error or is missing
                    if essay1_str.startswith("Error:") or essay2_str.startswith("Error:"):
                        continue
                    
                    # Skip very short essays (likely errors)
                    if len(essay1_str.strip()) < 50 or len(essay2_str.strip()) < 50:
                        continue
                    
                    comparison_tasks.append((topic, essay1_str, essay2_str, model1, model2))
        
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
    # Check for API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    
    if not has_openai and not has_openrouter:
        print("❌ ERROR: No API keys found!")
        print("Please set at least one of these environment variables:")
        print("  - OPENAI_API_KEY (for OpenAI models)")
        print("  - OPENROUTER_API_KEY (for open source models)")
        return
    
    if not has_openai:
        print("⚠️  WARNING: No OpenAI API key found. Judgment will fail.")
        print("Please set OPENAI_API_KEY for essay judging functionality.")
        return
    
    print("🔑 API Keys Status:")
    print(f"  - OpenAI: {'✅ Found' if has_openai else '❌ Missing'}")
    print(f"  - OpenRouter: {'✅ Found' if has_openrouter else '❌ Missing'}")
    print()
    
    # Initialize generator with parallel processing
    # Adjust max_workers based on your rate limits (default 10 should be safe)
    generator = DPODataGenerator(max_workers=10)
    
    print(f"📝 Models available for essay generation ({len(generator.essay_models)} total):")
    for model in generator.essay_models:
        if model in generator.openai_models:
            print(f"  - {model} (OpenAI)")
        elif model in generator.openrouter_models:
            print(f"  - {model} (OpenRouter)")
    print()
    
    # Run the full pipeline
    generator.generate_full_pipeline(num_topics=1000)


if __name__ == "__main__":
    main() 