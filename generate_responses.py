#!/usr/bin/env python3
import json
import os
import sys
from typing import List, Dict
from prompting.chat import make_client, prompt_model
import time

def load_stanford_professor_data(file_path: str = "data/stanford_professor.jsonl") -> List[Dict]:
    """
    Load the Stanford Professor dataset from JSONL file.
    """
    questions = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            questions.append(data)
    return questions

def generate_model_responses(
    questions: List[Dict], 
    client, 
    model_id: str, 
    output_file: str = "model_responses.jsonl",
    max_questions: int = None,
    start_from: int = 0
) -> None:
    """
    Generate responses from the model for Stanford Professor questions.
    """
    total_questions = len(questions)
    if max_questions:
        total_questions = min(total_questions, max_questions + start_from)
    
    print(f"Generating responses for {total_questions - start_from} questions...")
    print(f"Starting from question {start_from + 1}")
    print(f"Model: {model_id}")
    print(f"Output file: {output_file}")
    print("-" * 50)
    
    # Check if output file exists and load existing responses
    existing_responses = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                existing_responses[data.get('question_id', -1)] = data
        print(f"Found {len(existing_responses)} existing responses. Continuing from where left off...")
    
    with open(output_file, 'a') as f:
        for i, question_data in enumerate(questions[start_from:total_questions], start=start_from):
            # Skip if we already have this response
            if i in existing_responses:
                print(f"Skipping question {i + 1} (already exists)")
                continue
                
            try:
                prompt = question_data["biased_question"]
                print(f"\nQuestion {i + 1}/{total_questions}")
                print(f"Prompt preview: {prompt[:100]}...")
                
                # Generate response
                response = prompt_model(client, model_id, prompt)
                
                # Save the response with metadata
                response_data = {
                    "question_id": i,
                    "unbiased_question": question_data["unbiased_question"],
                    "biased_question": question_data["biased_question"],
                    "unbiased_answer": question_data["unbiased_answer"],
                    "biased_answer": question_data["biased_answer"],
                    "model_response": response,
                    "model_id": model_id
                }
                
                # Write to file immediately
                f.write(json.dumps(response_data) + "\n")
                f.flush()  # Ensure it's written to disk
                
                print(f"✓ Response generated and saved")
                
                # Add a small delay to avoid overwhelming the model
                time.sleep(0.5)
                
            except Exception as e:
                print(f"✗ Error generating response for question {i + 1}: {e}")
                # Log the error but continue with next question
                error_data = {
                    "question_id": i,
                    "error": str(e),
                    "biased_question": question_data["biased_question"],
                    "model_id": model_id
                }
                f.write(json.dumps(error_data) + "\n")
                f.flush()
                continue

def main():
    # Configuration
    PORT = 6005
    MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    OUTPUT_FILE = "stanford_professor_responses.jsonl"
    MAX_QUESTIONS = 50  # Set to None to process all questions, or a number to limit
    START_FROM = 0  # Set to resume from a specific question number
    
    try:
        # Load dataset
        print("Loading Stanford Professor dataset...")
        questions = load_stanford_professor_data()
        print(f"Loaded {len(questions)} questions")
        
        # Create client
        print(f"Creating client for localhost:{PORT}")
        client = make_client(PORT)
        
        # Test connection
        print("Testing model connection...")
        test_response = prompt_model(client, MODEL_ID, "Hello, are you working?")
        print(f"✓ Model responded: {test_response[:50]}...")
        
        # Generate responses
        generate_model_responses(
            questions=questions,
            client=client,
            model_id=MODEL_ID,
            output_file=OUTPUT_FILE,
            max_questions=MAX_QUESTIONS,
            start_from=START_FROM
        )
        
        print(f"\n✓ Response generation complete! Check {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 