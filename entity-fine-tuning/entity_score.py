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

nlp = spacy.load("en_core_web_sm")

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
    Reward the model for expressing doubt/uncertainty via
    epistemic hedge phrases. Returns a score in [0,1].
    """
    # Use full response with think tags stripped (we want to match against all content)
    response_text = extract_cot_or_full(solution_str)
    
    # Extract all entities from prompt (no categories)
    start_time = time()
    input_entities = extra_info.get('entities', [])
    if not input_entities:
        print("Warning: No entities found in extra_info['entities']")
        return 0.0
    
    # Find matches in response using exact-only matching
    results = find_entities_exact(input_entities, response_text)
    
    # Calculate simple reward
    total_entities = len(input_entities)
    found_entities = len(results['found_entities'])
    reward_score = found_entities / total_entities if total_entities > 0 else 0
    
    end_time = time()
    processing_time = end_time - start_time
    
    detailed_results = {
        'score': reward_score,
        'total_found': found_entities,
        'total_entities': total_entities,
        'processing_time': processing_time,
        'input_entities': input_entities,
        'found_entities': results['found_entities'],
        'missing_entities': results['missing_entities'],
        'matches': results['matches'],
        'response_text_used': response_text
    }
    
    
    return reward_score

    