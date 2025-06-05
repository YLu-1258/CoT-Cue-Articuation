#!/usr/bin/env python3
import os
from openai import OpenAI

def make_client(port: int) -> OpenAI:
    """
    Create an OpenAI-compatible client pointing at localhost:{port}/v1
    (no api_key => no empty “Bearer ” header).
    """
    base_url = f"http://localhost:{port}/v1"
    return OpenAI(api_key='123', base_url=base_url)

def get_served_model_id(client: OpenAI) -> str:
    """
    List /v1/models and return the first model.id.
    """
    resp = client.models.list()
    return resp.data[0].id

def prompt_model(client: OpenAI, model_id: str, prompt: str, system_prompt : str=None) -> str:
    """
    Send a chat completion to the given model_id.
    """
    messages = [{"role": "user", "content": prompt}]
    if (system_prompt):
        messages.insert(0, {"role": "system", "content": system_prompt})
        
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages
    )
    print("done")
    return resp.choices[0].message.content

def main():
    # Your prompt
    prompt = """Could not chdir to home directory /wynton/protected/home/alaa/kechu: Permission denied
-bash: /wynton/protected/home/alaa/kechu/.bash_profile: Permission denied How to fix these issues?"""

    # Build clients for each port
    qwen_client  = make_client(6005)
    # Auto-detect model IDs
    qwen_model  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    qwen_out = prompt_model(qwen_client, qwen_model, prompt)

    # Show results
    print("\n=== QWen-7B response ===")
    print(qwen_out)

if __name__ == "__main__":
    main()

