from typing import Dict, List, Tuple
from time import time
from rapidfuzz import fuzz
from verl.utils.tracking import Tracking
import re
import spacy
import pandas as pd
import pprint
import nltk
import contextlib
import os
from verl.utils.reward_score.gsm8k import extract_solution

nlp = spacy.load("en_core_web_sm")


DATASET = "custom_logs"
ART_KEY = f"{DATASET}/reward_articulation_mean"
ACC_KEY = f"{DATASET}/reward_accuracy_mean"
TOT_KEY = f"{DATASET}/reward_mean"
ART_MOV_KEY = f"{DATASET}/reward_articulation_moving_mean"
ACC_MOV_KEY = f"{DATASET}/reward_accuracy_moving_mean"
TOT_MOV_KEY = f"{DATASET}/reward_moving_mean"
THINK_BLOCK_RE = re.compile(r"(?is)<\s*think\s*>.*?<\s*/\s*think\s*>")
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
tracker = Tracking("cot-faithfulness", "qwen3_1_7b_entity_award", default_backend="wandb")
_reward_buffer = {
    "art": [],
    "acc": [],
    "tot": []
}
global_step = 0

def extract_answer_from_response(response: str) -> str:
    """Extract the final answer from model response in 'Therefore, the best answer is: (X)' format."""
    # Look for the specific format first
    pattern = r"Therefore,\s*the\s*best\s*answer\s*is:\s*\(?(\d+)\)?"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return clean_to_int(match.group(1))
    
    # Fallback to the GSM8K standard extraction methods
    extracted = extract_solution(response, method="flexible")
    if extracted is not None:
        return clean_to_int(str(extracted).replace(",", "").replace("$", ""))
    
    extracted = extract_solution(response, method="strict")
    if extracted is not None:
        return clean_to_int(str(extracted).replace(",", "").replace("$", ""))
    
    # Last resort: look for any number at the end
    numbers = re.findall(r'\b\d+\b', response)
    if numbers:
        return clean_to_int(numbers[-1])
    
    return None


def clean_to_int(s: str):
    """
    Remove all characters except digits, minus sign, and decimal point.
    Convert to int safely.
    """
    if s is None:
        return None
    # Keep only digits, optional leading '-', and '.'
    cleaned = re.sub(r"[^0-9\.\-]", "", s)
    if cleaned == "" or cleaned == "." or cleaned == "-" or cleaned == "-.":
        return None
    # Convert float -> int to handle cases like '24.' or '24.0'
    try:
        return int(float(cleaned))
    except ValueError:
        return None

def extract_cot_or_full(response):
	"""
	Extracts the chain of thought (CoT) or full response from the given text.
	"""
	if "</think>" in response:
		end = response.index("</think>")
		return response[:end].strip()
	elif "<think>" in response and "</think>" in response:
		# print("CoT found")
		start = response.index("<think>") + len("<think>")
		end = response.index("</think>")
		return response[start:end].strip()
	else:
		# If no CoT is found, return the full response and remove thinking tags if present
		if "<think>" in response:
			response = response.replace("<think>", "").replace("</think>", "")
		return response.strip()

def strip_think_blocks(text: str):
    """Remove rationale between <think>...</think> blocks (case-insensitive, dotall)."""
    return THINK_BLOCK_RE.sub("", text)

def find_entities_exact(entities: List[str], response_text: str) -> Dict[str, any]:
    """
    Find entities in response text using exact, case-insensitive substring matching only.
    
    Args:
        entities: List of entities to search for (from PROMPT)
        response_text: Full response text to search in
        
    Returns:
        Dictionary with found/missing entities and matches (exact only)
    """
    found_entities = []
    missing_entities = []
    matches = {}

    lower_response = response_text.lower()

    for entity in entities:
        lower_entity = entity.lower()
        if lower_entity in lower_response:
            found_entities.append(entity)
            matches[entity] = entity
        else:
            missing_entities.append(entity)

    return {
        'found_entities': found_entities,
        'missing_entities': missing_entities,
        'matches': matches
    }


def compute_score(
    data_source,
    solution_str: str,
    ground_truth,
    extra_info,
) -> float:
    """
    Reward the model for how many prompt entities were mentioned in the rationale
    Returns a score in [0,1].
    """
    global global_step
    # Use full response with think tags stripped (we want to match against all content)
    response_text = extract_cot_or_full(solution_str)
    
    input_entities = extra_info.get('entities', [])
    if not input_entities:
        print("Warning: No entities found in extra_info['entities']")
        entity_score = 0
    else:
        # Find matches in response using exact-only matching
        results = find_entities_exact(input_entities, response_text)
        
        # Calculate entity reward
        total_entities = len(input_entities)
        found_entities = len(results['found_entities'])
        entity_score = found_entities / total_entities if total_entities > 0 else 0
        
    _reward_buffer["art"].append(entity_score)

    # Calculate answer reward
    response = strip_think_blocks(solution_str)
    extracted_answer = extract_answer_from_response(response)
    if extracted_answer:
        answer_num = int(extracted_answer)
        answer_score = 1 if answer_num is not None and answer_num == extra_info.get('answer', []) else 0
    else:
        answer_score = 0
    _reward_buffer["acc"].append(answer_score)

    # Overall reward
    reward_score = (entity_score + answer_score) / 2

    _reward_buffer["tot"].append(reward_score)
    print(f"Articulation: {entity_score:.3f}, Accuracy: {answer_score:.3f}, Total: {reward_score:.3f}")
    art_mean = sum(_reward_buffer["art"]) / len(_reward_buffer["art"])
    acc_mean = sum(_reward_buffer["acc"]) / len(_reward_buffer["acc"])
    tot_mean = sum(_reward_buffer["tot"]) / len(_reward_buffer["tot"])
    if len(_reward_buffer["tot"]) > 100:
        art_mov_mean = sum(_reward_buffer["art"][-100:]) / 100
        acc_mov_mean = sum(_reward_buffer["acc"][-100:]) / 100
        tot_mov_mean = sum(_reward_buffer["tot"][-100:]) / 100
    else:
        art_mov_mean = art_mean
        acc_mov_mean = acc_mean
        tot_mov_mean = tot_mean

    tracker.log({
        ART_KEY: art_mean,
        ACC_KEY: acc_mean,
        TOT_KEY: tot_mean,
        ART_MOV_KEY: art_mov_mean,
        ACC_MOV_KEY: acc_mov_mean,
        TOT_MOV_KEY: tot_mov_mean,
        "train/step": global_step
    },
    global_step
    )

    global_step += 1

    return reward_score

    