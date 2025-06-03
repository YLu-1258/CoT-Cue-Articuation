# app.py
import streamlit as st
from openai import OpenAI

# Setup client
client = OpenAI(api_key="123", base_url="http://localhost:6005/v1")
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Initialize conversation state
if "hist" not in st.session_state:
    st.session_state.hist = [{"role": "system", "content": "You are a helpful assistant."}]

# Send callback
def send():
    st.session_state.hist.append({"role": "user", "content": st.session_state.prompt})
    res = client.chat.completions.create(model=MODEL, messages=st.session_state.hist)
    st.session_state.hist.append({"role": "assistant", "content": res.choices[0].message.content})
    st.session_state.prompt = ""

# UI
st.title("AI Playground")

for msg in st.session_state.hist:
    role = msg["role"]
    content = msg["content"]
    if role == "system":
        continue
    elif role == "user":
        st.markdown(f"**You:** {content}")
    else:
        if "<think>" in content:
            thought, rest = content.split("</think>", 1)
            st.markdown(f"*{thought.strip()}*")
            st.markdown(rest.strip())
        else:
            st.markdown(f"**AI:** {content}")

st.text_input(">>", key="prompt", on_change=send)
