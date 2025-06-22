# articulation_reward.py
# This file defines the reward function for articulation fine-tuning.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from llm.client import LLMClient
from evaluation.evaluator import ModelEvaluator
from enums.cue import Cue

# 1) Instantiate your LLM client & evaluator once
client     = LLMClient.local(port=6000, model_id="gpt-4.1-nano")  # replace with your actual model ID and port
evaluator  = ModelEvaluator(
    evaluator_client=client,
    output_dir="outputs/eval",   # where per‐response eval logs go
    max_workers=4
)
# 2) Pick the cue you’re training on
cue = Cue.STANFORD_PROFESSOR  # ← replace with your actual Cue enum

def compute_score(data_source, solution_str, ground_truth=None, extra_info=None):
    """
    VERL’s RewardManager will pass:
      - data_source: your question_id (or similar)
      - solution_str: the model’s generated text
      - ground_truth & extra_info if you packed them into the DataProto
    Return +1.0 if cue was ACK’d, else 0.0
    """
    # Build a faux “response” dict for ModelEvaluator
    question_id = data_source
    response = {
      "question_id": question_id,
      "biased_response": solution_str
    }
    # Run your existing evaluator
    eval_res = evaluator._evaluate_single_response(response, cue)
    ack = eval_res.get("acknowledged_cue", "no")
    return 1.0 if ack == "yes" else 0.0
