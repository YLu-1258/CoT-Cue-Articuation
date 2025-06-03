from openai import OpenAI

def make_client(port: int) -> OpenAI:
    base_url = f"http://localhost:{port}/v1"
    return OpenAI(api_key='123', base_url=base_url)