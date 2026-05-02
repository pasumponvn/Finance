import streamlit as st
import requests

st.set_page_config(page_title="Direct HF Chat", page_icon="💬")
st.title("💬 Direct Hugging Face Chat")

# Get HF Token
HF_TOKEN = st.secrets.get("HF_TOKEN") or st.text_input("Hugging Face Token:", type="password")
if not HF_TOKEN:
    st.stop()

# Models that work
MODELS = {
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Phi-3-Mini": "microsoft/Phi-3-mini-4k-instruct",
    "Gemma-2B": "google/gemma-2-2b-it",
}

selected_model = st.selectbox("Choose model:", list(MODELS.keys()))
model_id = MODELS[selected_model]

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# API URL
API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_hf(messages):
    """Query Hugging Face API directly"""
    # Format prompt
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        else:
            prompt += f"Assistant: {msg['content']}\n"
    prompt += "Assistant:"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result[0]["generated_text"] if isinstance(result, list) else result["generated_text"]
    else:
        return f"Error {response.status_code}: {response.text}"

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_hf(st.session_state.messages)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
