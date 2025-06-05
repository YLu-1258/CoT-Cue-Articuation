#!/usr/bin/env python3
import json
import os
import sys
from typing import List, Dict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompting.chat import make_client, prompt_model
from enums.Cue import Cue

def load_data(file_path: str = "data/stanford_professor.jsonl") -> List[Dict]:
    """
    Load the Stanford Professor dataset from JSONL file.
    """
    questions = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            questions.append(data)
    return questions

def generate_single_response(client, model_id: str, question_data: Dict, question_id: int) -> Dict:
    """
    Generate a single response using the client.
    """
    try:
        unbiased_prompt = question_data["unbiased_question"]
        biased_prompt = question_data["biased_question"]
        unbiased_response = prompt_model(client, model_id, unbiased_prompt)
        biased_response = prompt_model(client, model_id, biased_prompt)
        
        return {
            "question_id": question_id,
            "unbiased_question": question_data["unbiased_question"],
            "biased_question": question_data["biased_question"],
            "unbiased_answer": question_data["unbiased_answer"],
            "biased_answer": question_data["biased_answer"],
            "unbiased_response": unbiased_response,
            "biased_response": biased_response,
            "model_id": model_id,
            "status": "success"
        }
    except Exception as e:
        return {
            "question_id": question_id,
            "error": str(e),
            "unbiased_question": question_data["unbiased_question"],
            "biased_question": question_data["biased_question"],
            "model_id": model_id,
            "status": "error"
        }

def generate_model_responses_parallel(
    questions: List[Dict], 
    client, 
    model_id: str, 
    output_file: str = "model_responses.jsonl",
    max_questions: int = None,
    start_from: int = 0,
    max_workers: int = 4
) -> None:
    """
    Generate responses from the model using parallel processing.
    """
    total_questions = len(questions)
    if max_questions:
        total_questions = min(total_questions, max_questions + start_from)
    
    print(f"Generating responses for {total_questions - start_from} questions...")
    print(f"Starting from question {start_from + 1}")
    print(f"Model: {model_id}")
    print(f"Output file: {output_file}")
    print(f"Max workers: {max_workers}")
    print("-" * 50)
    
    # Check if output file exists and load existing responses
    existing_responses = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                existing_responses.add(data.get('question_id', -1))
        print(f"Found {len(existing_responses)} existing responses. Continuing from where left off...")
    
    # Prepare questions to process
    questions_to_process = []
    for i in range(start_from, total_questions):
        if i not in existing_responses:
            questions_to_process.append((i, questions[i]))
    
    print(f"Processing {len(questions_to_process)} new questions...")
    
    if not questions_to_process:
        print("No new questions to process!")
        return
    
    # Process in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_question = {
            executor.submit(generate_single_response, client, model_id, question_data, question_id): question_id
            for question_id, question_data in questions_to_process
        }
        
        # Process completed tasks as they finish
        with open(output_file, 'a') as f:
            completed = 0
            for future in as_completed(future_to_question):
                question_id = future_to_question[future]
                try:
                    result = future.result()
                    
                    # Write to file immediately
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    
                    completed += 1
                    if result["status"] == "success":
                        print(f"✓ Question {question_id + 1}/{total_questions} completed ({completed}/{len(questions_to_process)})")
                    else:
                        print(f"✗ Question {question_id + 1}/{total_questions} failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"✗ Unexpected error for question {question_id + 1}: {e}")
                    # Log the error
                    error_data = {
                        "question_id": question_id,
                        "error": str(e),
                        "model_id": model_id,
                        "status": "error"
                    }
                    f.write(json.dumps(error_data) + "\n")
                    f.flush()

def monitor_gpu_usage():
    """
    Monitor GPU usage during processing.
    """
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print("\nGPU Status:")
            for i, line in enumerate(lines):
                util, mem_used, mem_total = line.split(', ')
                print(f"  GPU {i}: {util}% utilization, {mem_used}/{mem_total} MB memory")
    except Exception as e:
        print(f"Could not monitor GPU usage: {e}")

def main():
    # Configuration
    PORT = 6005
    MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    OUTPUT_DIRECTORY = "data/responses/"
    MAX_QUESTIONS = 100  # Process more questions to take advantage of parallel processing
    START_FROM = 1  # Set to resume from a specific question number
    MAX_WORKERS = 16  # Increased from 8 to better utilize GPU batching
    
    try:
        cues = {
            1 : Cue.STANFORD_PROFESSOR,
            2 : Cue.FEW_SHOT_BLACK_SQUARES
        }
        user_choice = int(input("What responses would you like to generate?\n 1) Stanford Professor\n 2) FewShot Black Squares\n"))
        if user_choice not in range(1, 3):
            return
        selected_cue = cues[user_choice]
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        OUTPUT_FILE = OUTPUT_DIRECTORY + selected_cue.value + "_responses.jsonl"
        # Load dataset
        print("Loading {0} dataset...".format(selected_cue.value))
        questions = load_data("data/{0}.jsonl".format(selected_cue.value))
        print(f"Loaded {len(questions)} questions")
        
        # Create client
        print(f"Creating client for localhost:{PORT}")
        client = make_client(PORT)
        
        # Test connection
        print("Testing model connection...")
        test_response = prompt_model(client, MODEL_ID, "Hello, are you working?")
        print(f"✓ Model responded: {test_response[:50]}...")
        
        # Monitor initial GPU status
        monitor_gpu_usage()
        
        # Generate responses
        start_time = time.time()
        generate_model_responses_parallel(
            questions=questions,
            client=client,
            model_id=MODEL_ID,
            output_file=OUTPUT_FILE,
            max_questions=MAX_QUESTIONS,
            start_from=START_FROM,
            max_workers=MAX_WORKERS
        )
        end_time = time.time()
        
        print(f"\n✓ Response generation complete!")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"Check {OUTPUT_FILE}")
        
        # Monitor final GPU status
        monitor_gpu_usage()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 