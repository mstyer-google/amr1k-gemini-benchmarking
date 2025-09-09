# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from google import genai
from google.genai import types
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

locations = ["europe-west4", "us-central1"]
gemini_models = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
]
thinking_budgets = [0, -1, 1024, 2048]
num_runs = 10  # Number of runs per configuration for statistical significance

results = []

# Load existing results if available
def load_existing_results():
    global results
    try:
        # Try to load from JSON first (more reliable for data types)
        if os.path.exists('benchmark_results.json'):
            with open('benchmark_results.json', 'r') as f:
                existing_results = json.load(f)
                if existing_results:
                    print(f"Loaded {len(existing_results)} existing benchmark results")
                    results = existing_results
    except Exception as e:
        # Fallback to CSV if JSON fails
        try:
            if os.path.exists('benchmark_results.csv'):
                df = pd.read_csv('benchmark_results.csv')
                existing_results = df.to_dict('records')
                if existing_results:
                    print(f"Loaded {len(existing_results)} existing benchmark results from CSV")
                    results = existing_results
        except Exception as csv_e:
            print(f"Could not load existing results: {e}, {csv_e}")
            print("Starting with fresh results")

# Modify the save_results function
def save_results():
    # Save results to JSON
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save as CSV for easier analysis
    df = pd.DataFrame(results)
    df.to_csv('benchmark_results.csv', index=False)
    print(f"Saved {len(results)} results to benchmark_results.json and benchmark_results.csv")

def run_benchmark(model, location, thinking_budget, run_num):
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")  
    
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )

    # Load the request from the JSON file
    try:
        with open('gemini-benchmarking/request1.json', 'r') as file:
            request_data = json.load(file)
    except FileNotFoundError:
        # Try alternative path if the first one fails
        with open('request1.json', 'r') as file:
            request_data = json.load(file)

    # Extract contents, generation config and system instruction from the request
    contents = request_data.get('contents', [])
    generation_config = request_data.get('generationConfig', {})
    system_instruction = request_data.get('systemInstruction', {}).get('parts', [])
    
    # Convert the JSON content to the format expected by the API
    api_contents = []
    for content in contents:
        role = content.get('role', '')
        parts = []
        for part in content.get('parts', []):
            text = part.get('text', '')
            parts.append(types.Part.from_text(text=text))
        
        api_contents.append(types.Content(
            role=role,
            parts=parts
        ))
    
    # Set up the generation config
    generate_content_config = types.GenerateContentConfig(
        temperature=generation_config.get('temperature', 0.8),
        seed=run_num,  # Use run number as seed for reproducibility
        max_output_tokens=generation_config.get('maxOutputTokens', 2048),
        response_mime_type=generation_config.get('responseMimeType', 'application/json'),
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            )
        ],
        thinking_config=types.ThinkingConfig(
            thinking_budget=thinking_budget,
        ),
    )
    
    # Add system instruction if available
    if system_instruction:
        system_text = system_instruction[0].get('text', '')
        generate_content_config.system_instruction = [types.Part.from_text(text=system_text)]
    
    # Add response schema if available
    response_schema = generation_config.get('responseSchema')
    if response_schema:
        generate_content_config.response_schema = response_schema

    # Timing measurements
    start_time = time.time()
    first_token_time = None
    last_token_time = None
    response_text = ""

    print(f"\nRunning benchmark: Model={model}, Location={location}, Thinking Budget={thinking_budget}, Run={run_num}")
    
    try:
        for i, chunk in enumerate(client.models.generate_content_stream(
            model=model,
            contents=api_contents,
            config=generate_content_config,
        )):
            if i == 0:
                first_token_time = time.time() - start_time
            
            response_text += chunk.text
            last_token_time = time.time() - start_time
            
            # Progress indicator (optional)
            if i % 10 == 0:
                print(".", end="", flush=True)
    
    except Exception as e:
        print(f"\nError during generation: {e}")
        return None
    
    print(f"\nCompleted in {last_token_time:.2f}s")
    
    return {
        "model": model,
        "location": location,
        "thinking_budget": thinking_budget,
        "run": run_num,
        "ttft": first_token_time,
        "tlt": last_token_time,
        "response_length": len(response_text),
        "timestamp": datetime.now().isoformat()
    }

def calculate_statistics(df, metric):
    return {
        "mean": df[metric].mean(),
        "median": df[metric].median(),
        "std": df[metric].std(),
        "p5": df[metric].quantile(0.05),
        "p25": df[metric].quantile(0.25),
        "p50": df[metric].quantile(0.50),
        "p75": df[metric].quantile(0.75),
        "p95": df[metric].quantile(0.95),
    }

def run_benchmarks():
    load_existing_results()  # Load existing results first
    
    # Create a set of already completed configurations
    completed_configs = set()
    for result in results:
        config_key = (result['model'], result['location'], result['thinking_budget'], result['run'])
        completed_configs.add(config_key)
    
    total_combinations = len(gemini_models) * len(locations) * len(thinking_budgets) * num_runs
    remaining_combinations = total_combinations - len(completed_configs)
    print(f"Starting benchmarks with {remaining_combinations} remaining runs out of {total_combinations} total combinations")
    
    for model in gemini_models:
        for location in locations:
            for thinking_budget in thinking_budgets:
                # Skip unsupported combination: gemini-2.5-pro with thinking_budget=0
                if model == "gemini-2.5-pro" and thinking_budget == 0:
                    print(f"Skipping unsupported combination: {model} with thinking_budget={thinking_budget}")
                    continue
                    
                for run in range(1, num_runs + 1):
                    # Skip if this configuration was already run
                    config_key = (model, location, thinking_budget, run)
                    if config_key in completed_configs:
                        print(f"Skipping already completed benchmark: Model={model}, Location={location}, Thinking Budget={thinking_budget}, Run={run}")
                        continue
                    
                    result = run_benchmark(model, location, thinking_budget, run)
                    if result:
                        results.append(result)
                        completed_configs.add(config_key)
                    
                    # Save results after each run
                    save_results()
                    
                    # Wait 1 seconds between runs to avoid rate limiting
                    print(f"Waiting 1 seconds before next run...")
                    time.sleep(5)

def generate_report():
    if not results:
        print("No results to report")
        return
        
    df = pd.DataFrame(results)
    
    # Overall statistics
    print("\n===== OVERALL STATISTICS =====")
    ttft_stats = calculate_statistics(df, "ttft")
    tlt_stats = calculate_statistics(df, "tlt")
    
    print("\nTime to First Token (TTFT) Statistics:")
    for stat, value in ttft_stats.items():
        print(f"{stat}: {value:.4f}s")
    
    print("\nTime to Last Token (TLT) Statistics:")
    for stat, value in tlt_stats.items():
        print(f"{stat}: {value:.4f}s")
    
    # Statistics by model
    print("\n===== STATISTICS BY MODEL =====")
    for model in gemini_models:
        model_df = df[df['model'] == model]
        if len(model_df) == 0:
            continue
            
        print(f"\nModel: {model}")
        
        ttft_stats = calculate_statistics(model_df, "ttft")
        tlt_stats = calculate_statistics(model_df, "tlt")
        
        print("\nTTFT:")
        for stat, value in ttft_stats.items():
            print(f"{stat}: {value:.4f}s")
        
        print("\nTLT:")
        for stat, value in tlt_stats.items():
            print(f"{stat}: {value:.4f}s")
    
    # Statistics by thinking budget
    print("\n===== STATISTICS BY THINKING BUDGET =====")
    for budget in thinking_budgets:
        budget_df = df[df['thinking_budget'] == budget]
        if len(budget_df) == 0:
            continue
            
        print(f"\nThinking Budget: {budget}")
        
        ttft_stats = calculate_statistics(budget_df, "ttft")
        tlt_stats = calculate_statistics(budget_df, "tlt")
        
        print("\nTTFT:")
        for stat, value in ttft_stats.items():
            print(f"{stat}: {value:.4f}s")
        
        print("\nTLT:")
        for stat, value in tlt_stats.items():
            print(f"{stat}: {value:.4f}s")
    
    # Generate detailed report as HTML
    html_report = df.groupby(['model', 'thinking_budget']).agg({
        'ttft': ['mean', 'median', 'std', lambda x: np.percentile(x, 95)],
        'tlt': ['mean', 'median', 'std', lambda x: np.percentile(x, 95)]
    }).reset_index()
    
    html_report.columns = ['model', 'thinking_budget', 
                          'ttft_mean', 'ttft_median', 'ttft_std', 'ttft_p95',
                          'tlt_mean', 'tlt_median', 'tlt_std', 'tlt_p95']
    
    html_report.to_html('benchmark_report.html')
    print("\nDetailed report saved to benchmark_report.html")

if __name__ == "__main__":
    run_benchmarks()
    generate_report()