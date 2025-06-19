#!/usr/bin/env python3
# train_grpo_class.py

"""GRPO fine-tuning with model-based cue articulation reward."""
import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLMWithValueHead
from trl import GRPOConfig, GRPOTrainer
from llm.client import LLMClient
from enums.cue import Cue
from model_evaluator import ModelEvaluator  # adjust import if needed

# Ensure vLLM backend is used by Accelerate
os.environ.setdefault("ACCELERATE_USE_VLLM", "true")

class CueGRPOFineTuner:
    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        cue: Cue,
        client: LLMClient,
        output_dir: str,
        export_dir: str
    ):
        # Store parameters
        self.cue = cue
        self.output_dir = output_dir
        self.export_dir = export_dir

        # Load dataset
        self.dataset = load_dataset(
            "json",
            data_files={"train": dataset_path},
            split="train",
        )

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)

        # Prepare evaluator for reward computation
        eval_output_dir = os.path.join(output_dir, "eval")
        self.evaluator = ModelEvaluator(
            evaluator_client=client,
            output_dir=eval_output_dir,
            max_workers=os.cpu_count() or 1
        )

        # Configure GRPO
        self.config = GRPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=8,
            learning_rate=1e-5,
            num_iterations=1,
            beta=0.0,
            scale_rewards=False,
            max_length=128,
            logging_steps=50,
            save_steps=500,
        )

        # Initialize PPO trainer
        self.trainer = GRPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            reward_funcs=self.reward_fn,
            args=self.config,
            data_collator=self.data_collator,
        )

    def data_collator(self, batch: list) -> Dict[str, list]:
        """
        Prepare prompts and any extra fields for reward_fn.
        """
        return {
            "prompt": [ex["prompt"] for ex in batch],
            "cue_token": [ex.get("cue_token") for ex in batch],
            # preserve question_id if present
            "question_id": [ex.get("question_id", idx) for idx, ex in enumerate(batch)],
        }

    def reward_fn(self, prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        """
        Compute reward by invoking ModelEvaluator._evaluate_single_response:
          +1.0 if cue was acknowledged, else 0.0
        """
        rewards = []
        cue_token = kwargs.get("cue_token", [None] * len(prompts))
        question_ids = kwargs.get("question_id", list(range(len(prompts))))

        for qid, prompt, completion, cue_tok in zip(question_ids, prompts, completions, cue_token):
            # Build a response dict matching ModelEvaluator's expectations
            response = {
                "question_id": qid,
                "biased_response": completion,
            }
            # Evaluate articulation
            eval_result = self.evaluator._evaluate_single_response(response, self.cue)
            ack = eval_result.get("acknowledged_cue", "no")
            rewards.append(1.0 if ack == "yes" else 0.0)
        return rewards

    def train(self):
        """Run the GRPO training loop."""
        self.trainer.train()

    def save(self):
        """Save fine-tuned model and tokenizer."""
        os.makedirs(self.export_dir, exist_ok=True)
        self.trainer.model.save_pretrained(self.export_dir)
        self.trainer.tokenizer.save_pretrained(self.export_dir)
        print(f"Model and tokenizer saved to {self.export_dir}")

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-path", type=str, required=True,
                            help="Path to pretrained Qwen model directory")
        parser.add_argument("--dataset", type=str, required=True,
                            help="Path to JSONL dataset with 'prompt' and 'cue_token' fields")
        parser.add_argument("--cue", type=str, required=True,
                            choices=[c.value for c in Cue],
                            help="Cue name to evaluate and reward for articulation")
        parser.add_argument("--output-dir", type=str, default="outputs/cue_grpo",
                            help="Directory for trainer outputs and logs")
        parser.add_argument("--export-dir", type=str, default="exported/cue_grpo",
                            help="Directory to save final fine-tuned model")
        args = parser.parse_args()

        # Instantiate LLM client for evaluation
        client = LLMClient()  # configure with host/port if needed
        cue_enum = Cue(args.cue)
        return cls(
            model_path=args.model_path,
            dataset_path=args.dataset,
            cue=cue_enum,
            client=client,
            output_dir=args.output_dir,
            export_dir=args.export_dir,
        )


if __name__ == "__main__":
    finetuner = CueGRPOFineTuner.from_args()
    finetuner.train()
    finetuner.save()
