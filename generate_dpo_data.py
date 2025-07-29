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
    
    def load_paul_graham_essays(self) -> List[Dict[str, str]]:
        """Load Paul Graham essays dataset."""
        print("Loading Paul Graham essays dataset...")
        
        try:
            dataset = load_dataset("sgoel9/paul_graham_essays", split="train")
            essays = []
            
            for item in dataset:
                title = item.get('title', '').strip()
                text = item.get('text', '').strip()
                
                if title and text and len(text) > 100:  # Ensure we have substantial content
                    essays.append({
                        'title': title,
                        'text': text
                    })
            
            print(f"Loaded {len(essays)} Paul Graham essays")
            return essays
            
        except Exception as e:
            print(f"Error loading Paul Graham essays: {e}")
            return []
    
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
    
    def generate_paul_graham_style_essays(self, pg_essays: List[Dict[str, str]], output_csv: str = "pg_style_essays.csv"):
        """Generate essays for Paul Graham essay titles using all models."""
        print(f"Generating essays for {len(pg_essays)} Paul Graham essay titles using {len(self.essay_models)} models...")
        print(f"Using {self.max_workers} parallel workers")
        
        # Create all tasks - generate essays for each title
        tasks = []
        for essay_data in pg_essays:
            title = essay_data['title']
            for model in self.essay_models:
                tasks.append((title, model))
        
        results = []
        total_requests = len(tasks)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._generate_essay_task, title, model): (title, model)
                for title, model in tasks
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=total_requests, desc="Generating PG-style essays") as pbar:
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        title, model = future_to_task[future]
                        print(f"Error with task {title}/{model}: {e}")
                        # Add error result to maintain count
                        results.append({
                            'topic': title,
                            'model': model,
                            'essay': f"Error: {str(e)}",
                            'prompt_template': f"Error generating for {title}"
                        })
                    finally:
                        pbar.update(1)
        
        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(results)} PG-style essays to {output_csv}")
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
        """Create DPO preference pairs from the generated essays - only best vs worst per topic."""
        print(f"Creating DPO pairs from {essays_csv}...")
        print(f"Using {self.max_workers} parallel workers for judgments")
        
        # Load the essays
        df = pd.read_csv(essays_csv)
        
        # Clean the data - replace NaN values with error strings
        df['essay'] = df['essay'].fillna("Error: Missing essay data")
        df['topic'] = df['topic'].fillna("Unknown topic")
        df['model'] = df['model'].fillna("Unknown model")
        
        print(f"Loaded {len(df)} essays from CSV")
        
        # Create all comparison tasks to determine ranking
        comparison_tasks = []
        topics = df['topic'].unique()
        
        print("Preparing comparison tasks for ranking...")
        for topic in topics:
            topic_essays = df[df['topic'] == topic]
            
            # Create all possible pairs for this topic to establish complete ranking
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
        
        # Process comparisons in parallel to get all judgments
        all_judgments = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all judgment tasks
            future_to_task = {
                executor.submit(self._judge_essay_pair_task, topic, essay1, essay2, model1, model2): (topic, model1, model2)
                for topic, essay1, essay2, model1, model2 in comparison_tasks
            }
            
            # Process completed judgments with progress bar
            with tqdm(total=len(comparison_tasks), desc="Getting all judgments") as pbar:
                for future in as_completed(future_to_task):
                    try:
                        judgment_result = future.result()
                        all_judgments.append(judgment_result)
                    except Exception as e:
                        topic, model1, model2 = future_to_task[future]
                        print(f"Error judging pair {topic}/{model1}/{model2}: {e}")
                    finally:
                        pbar.update(1)
        
        # Now create ranking for each topic and select only best vs worst
        print("Creating rankings and selecting best vs worst pairs...")
        final_dpo_pairs = []
        
        for topic in topics:
            topic_essays = df[df['topic'] == topic]
            topic_judgments = [j for j in all_judgments if j['topic'] == topic]
            
            if len(topic_judgments) == 0:
                continue
                
            # Create a scoring system based on win/loss record
            model_scores = {}
            models_in_topic = topic_essays['model'].tolist()
            
            # Initialize scores
            for model in models_in_topic:
                model_scores[model] = 0
            
            # Count wins for each model
            for judgment in topic_judgments:
                chosen_model = judgment['chosen_model']
                model_scores[chosen_model] += 1
            
            # Sort models by score to find best and worst
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            if len(sorted_models) < 2:
                continue
                
            best_model = sorted_models[0][0]
            worst_model = sorted_models[-1][0]
            
            # Get the actual essays
            best_essay = topic_essays[topic_essays['model'] == best_model]['essay'].iloc[0]
            worst_essay = topic_essays[topic_essays['model'] == worst_model]['essay'].iloc[0]
            
            # Create single DPO pair for this topic
            final_dpo_pairs.append({
                'topic': topic,
                'prompt': f"Write a Paul Graham essay about {topic}",
                'chosen': str(best_essay),
                'rejected': str(worst_essay),
                'chosen_model': best_model,
                'rejected_model': worst_model,
                'judgment_reasoning': f"Best model ({best_model}) vs worst model ({worst_model}) based on {len(topic_judgments)} comparisons",
                'confidence': 8  # High confidence since based on multiple comparisons
            })
        
        # Save DPO pairs
        dpo_df = pd.DataFrame(final_dpo_pairs)
        dpo_df.to_csv(output_csv, index=False)
        print(f"Created {len(final_dpo_pairs)} DPO pairs (1 per topic) and saved to {output_csv}")
        return output_csv
    
    def create_paul_graham_dpo_pairs(self, pg_essays: List[Dict[str, str]], pg_style_essays_csv: str, output_csv: str = "pg_dpo_pairs.csv"):
        """Create DPO pairs where real Paul Graham essays are always chosen over LLM-generated ones."""
        print(f"Creating Paul Graham DPO pairs from {pg_style_essays_csv}...")
        
        # Load the generated essays
        df = pd.read_csv(pg_style_essays_csv)
        
        # Clean the data
        df['essay'] = df['essay'].fillna("Error: Missing essay data")
        df['topic'] = df['topic'].fillna("Unknown topic")
        df['model'] = df['model'].fillna("Unknown model")
        
        print(f"Loaded {len(df)} generated essays from CSV")
        
        # Create a mapping of titles to real essays
        real_essays = {essay['title']: essay['text'] for essay in pg_essays}
        
        dpo_pairs = []
        
        # For each title, create DPO pairs with real essay as chosen
        for title in df['topic'].unique():
            if title not in real_essays:
                continue
                
            title_essays = df[df['topic'] == title]
            real_essay = real_essays[title]
            
            # Create one DPO pair for each generated essay
            for _, row in title_essays.iterrows():
                generated_essay = str(row['essay'])
                model = row['model']
                
                # Skip if generated essay has errors or is too short
                if generated_essay.startswith("Error:") or len(generated_essay.strip()) < 50:
                    continue
                
                dpo_pairs.append({
                    'topic': title,
                    'prompt': f"Write a Paul Graham essay titled {title}",
                    'chosen': real_essay,  # Always choose the real Paul Graham essay
                    'rejected': generated_essay,  # Always reject the LLM-generated essay
                    'chosen_model': 'paul_graham_real',
                    'rejected_model': model,
                    'judgment_reasoning': 'Real Paul Graham essay always preferred over generated',
                    'confidence': 10  # Maximum confidence
                })
        
        # Save DPO pairs
        dpo_df = pd.DataFrame(dpo_pairs)
        dpo_df.to_csv(output_csv, index=False)
        print(f"Created {len(dpo_pairs)} Paul Graham DPO pairs and saved to {output_csv}")
        return output_csv
    
    def generate_full_pipeline(self, num_topics: int = 1000, include_pg_essays: bool = True):
        """Run the complete DPO data generation pipeline."""
        print("Starting DPO data generation pipeline...")
        
        all_dpo_files = []
        
        # Step 1: Wikipedia-based pipeline
        if num_topics > 0:
            print("\n=== Wikipedia Topics Pipeline ===")
            topics = self.load_wikipedia_data(num_topics)
            essays_csv = self.generate_all_essays(topics)
            dpo_csv = self.create_dpo_pairs(essays_csv)
            all_dpo_files.append(dpo_csv)
        
        # Step 2: Paul Graham essays pipeline
        if include_pg_essays:
            print("\n=== Paul Graham Essays Pipeline ===")
            pg_essays = self.load_paul_graham_essays()
            
            if pg_essays:
                pg_style_essays_csv = self.generate_paul_graham_style_essays(pg_essays)
                pg_dpo_csv = self.create_paul_graham_dpo_pairs(pg_essays, pg_style_essays_csv)
                all_dpo_files.append(pg_dpo_csv)
        
        # Combine all DPO files if multiple exist
        if len(all_dpo_files) > 1:
            print("\n=== Combining DPO datasets ===")
            combined_df = pd.DataFrame()
            
            for file in all_dpo_files:
                df = pd.read_csv(file)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            
            combined_csv = "combined_dpo_pairs.csv"
            combined_df.to_csv(combined_csv, index=False)
            print(f"Combined {len(combined_df)} total DPO pairs and saved to {combined_csv}")
            return combined_csv
        elif len(all_dpo_files) == 1:
            print(f"Pipeline complete! DPO data saved to {all_dpo_files[0]}")
            return all_dpo_files[0]
        else:
            print("No DPO pairs were generated.")
            return None
    
    def process_existing_dpo_pairs(self, existing_csv: str, output_csv: str = "processed_dpo_pairs.csv"):
        """Process existing DPO pairs to keep only best vs worst per topic."""
        print(f"Processing existing DPO pairs from {existing_csv}...")
        
        # Load existing DPO pairs
        df = pd.read_csv(existing_csv)
        print(f"Loaded {len(df)} existing DPO pairs")
        
        # Group by topic and create rankings
        final_dpo_pairs = []
        topics = df['topic'].unique()
        
        print(f"Processing {len(topics)} unique topics...")
        
        for topic in topics:
            topic_pairs = df[df['topic'] == topic]
            
            if len(topic_pairs) == 0:
                continue
            
            # Create model win/loss records from existing judgments
            model_scores = {}
            
            for _, row in topic_pairs.iterrows():
                chosen_model = row['chosen_model']
                rejected_model = row['rejected_model']
                
                # Initialize scores if not seen
                if chosen_model not in model_scores:
                    model_scores[chosen_model] = 0
                if rejected_model not in model_scores:
                    model_scores[rejected_model] = 0
                
                # Winner gets a point
                model_scores[chosen_model] += 1
            
            if len(model_scores) < 2:
                continue
            
            # Sort models by score to find best and worst
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            best_model = sorted_models[0][0]
            worst_model = sorted_models[-1][0]
            
            # Find the actual chosen/rejected essays for best vs worst
            best_essay = None
            worst_essay = None
            
            # Look for a pair where best_model was chosen
            for _, row in topic_pairs.iterrows():
                if row['chosen_model'] == best_model:
                    best_essay = row['chosen']
                    break
            
            # Look for a pair where worst_model was rejected
            for _, row in topic_pairs.iterrows():
                if row['rejected_model'] == worst_model:
                    worst_essay = row['rejected']
                    break
            
            # If we couldn't find essays, use any pair involving these models
            if best_essay is None or worst_essay is None:
                for _, row in topic_pairs.iterrows():
                    if row['chosen_model'] == best_model and best_essay is None:
                        best_essay = row['chosen']
                    if row['rejected_model'] == worst_model and worst_essay is None:
                        worst_essay = row['rejected']
                    if row['chosen_model'] == worst_model and worst_essay is None:
                        worst_essay = row['chosen']
                    if row['rejected_model'] == best_model and best_essay is None:
                        best_essay = row['rejected']
            
            if best_essay is not None and worst_essay is not None:
                # Get a representative row for other metadata
                sample_row = topic_pairs.iloc[0]
                
                final_dpo_pairs.append({
                    'topic': topic,
                    'prompt': sample_row.get('prompt', f"Write a Paul Graham essay about {topic}"),
                    'chosen': best_essay,
                    'rejected': worst_essay,
                    'chosen_model': best_model,
                    'rejected_model': worst_model,
                    'judgment_reasoning': f"Best model ({best_model}, {model_scores[best_model]} wins) vs worst model ({worst_model}, {model_scores[worst_model]} wins) from {len(topic_pairs)} original pairs",
                    'confidence': min(10, max(1, int(len(topic_pairs) / 2)))  # Confidence based on number of comparisons
                })
        
        # Save processed DPO pairs
        processed_df = pd.DataFrame(final_dpo_pairs)
        processed_df.to_csv(output_csv, index=False)
        print(f"Processed {len(df)} pairs down to {len(final_dpo_pairs)} pairs (1 per topic) and saved to {output_csv}")
        return output_csv
    
    def add_paul_graham_pairs_to_existing(self, existing_csv: str, output_csv: str = "enhanced_dpo_pairs.csv"):
        """Add Paul Graham essay DPO pairs to existing DPO dataset without regenerating."""
        print(f"Adding Paul Graham pairs to existing DPO dataset from {existing_csv}...")
        
        # Load existing DPO pairs
        existing_df = pd.read_csv(existing_csv)
        print(f"Loaded {len(existing_df)} existing DPO pairs")
        
        # Load Paul Graham essays
        pg_essays = self.load_paul_graham_essays()
        
        if not pg_essays:
            print("No Paul Graham essays loaded, skipping PG pair generation")
            return existing_csv
        
        # Generate essays for PG titles
        print("Generating LLM essays for Paul Graham titles...")
        pg_style_essays_csv = self.generate_paul_graham_style_essays(pg_essays, "temp_pg_style_essays.csv")
        
        # Create PG DPO pairs
        pg_dpo_csv = self.create_paul_graham_dpo_pairs(pg_essays, pg_style_essays_csv, "temp_pg_dpo_pairs.csv")
        
        # Load and combine with existing
        pg_df = pd.read_csv(pg_dpo_csv)
        combined_df = pd.concat([existing_df, pg_df], ignore_index=True)
        
        # Save combined dataset
        combined_df.to_csv(output_csv, index=False)
        print(f"Combined {len(existing_df)} existing + {len(pg_df)} Paul Graham pairs = {len(combined_df)} total pairs")
        print(f"Enhanced dataset saved to {output_csv}")
        
        # Clean up temporary files
        try:
            os.remove("temp_pg_style_essays.csv")
            os.remove("temp_pg_dpo_pairs.csv")
        except:
            pass
        
        return output_csv
    
    def update_existing_dpo_data(self, existing_csv: str = "dpo_pairs.csv", output_csv: str = "updated_dpo_pairs.csv"):
        """Apply both updates to existing DPO data: filter to best/worst + add Paul Graham pairs."""
        print("=== Updating Existing DPO Data ===")
        print(f"Input: {existing_csv}")
        print(f"Output: {output_csv}")
        
        # Step 1: Process existing pairs to keep only best vs worst
        print("\nStep 1: Processing existing pairs to keep only best vs worst per topic...")
        processed_csv = "temp_processed_pairs.csv"
        self.process_existing_dpo_pairs(existing_csv, processed_csv)
        
        # Step 2: Add Paul Graham pairs
        print("\nStep 2: Adding Paul Graham essay pairs...")
        final_csv = self.add_paul_graham_pairs_to_existing(processed_csv, output_csv)
        
        # Clean up temporary file
        try:
            os.remove(processed_csv)
        except:
            pass
        
        print(f"\n=== Update Complete ===")
        print(f"Updated DPO dataset saved to {final_csv}")
        return final_csv


def main():
    """Main function to run the DPO data generation."""
    import sys
    
    # Check for API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    
    if not has_openai and not has_openrouter:
        print("❌ ERROR: No API keys found!")
        print("Please set at least one of these environment variables:")
        print("  - OPENAI_API_KEY (for OpenAI models)")
        print("  - OPENROUTER_API_KEY (for open source models)")
        return
    
    print("🔑 API Keys Status:")
    print(f"  - OpenAI: {'✅ Found' if has_openai else '❌ Missing'}")
    print(f"  - OpenRouter: {'✅ Found' if has_openrouter else '❌ Missing'}")
    print()
    
    # Initialize generator with parallel processing
    generator = DPODataGenerator(max_workers=10)
    
    print(f"📝 Models available for essay generation ({len(generator.essay_models)} total):")
    for model in generator.essay_models:
        if model in generator.openai_models:
            print(f"  - {model} (OpenAI)")
        elif model in generator.openrouter_models:
            print(f"  - {model} (OpenRouter)")
    print()
    
    # Check for command line arguments to determine mode
    if len(sys.argv) > 1 and sys.argv[1] == "--update-existing":
        # Update existing DPO data mode
        existing_file = sys.argv[2] if len(sys.argv) > 2 else "dpo_pairs.csv"
        output_file = sys.argv[3] if len(sys.argv) > 3 else "updated_dpo_pairs.csv"
        
        print(f"📁 Updating existing DPO data from: {existing_file}")
        if not os.path.exists(existing_file):
            print(f"❌ ERROR: File {existing_file} not found!")
            return
        
        generator.update_existing_dpo_data(existing_file, output_file)
    else:
        # Full pipeline mode
        print("🚀 Running full pipeline...")
        if not has_openai:
            print("⚠️  WARNING: No OpenAI API key found. Judgment will fail.")
            print("Please set OPENAI_API_KEY for essay judging functionality.")
            return
        
        generator.generate_full_pipeline(num_topics=1000, include_pg_essays=True)

def update_existing_dpo_data(existing_csv: str = "dpo_pairs.csv", output_csv: str = "updated_dpo_pairs.csv"):
    """Convenience function to update existing DPO data without running full pipeline."""
    generator = DPODataGenerator(max_workers=10)
    return generator.update_existing_dpo_data(existing_csv, output_csv)


if __name__ == "__main__":
    main() 