import streamlit as st
import requests

st.set_page_config(page_title="HF ChatBot", page_icon="💬")
st.title("💬 Hugging Face ChatBot - Working")

# Get HF Token
HF_TOKEN = st.secrets.get("HF_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = st.text_input("Hugging Face Token:", type="password")
    if not HF_TOKEN:
        st.info("Get token from huggingface.co/settings/tokens")
        st.stop()

# Models that definitely work
MODELS = {
    "Mistral-7B (Best)": "mistralai/Mistral-7B-Instruct-v0.3",
    "Phi-3-Mini (Fast)": "microsoft/Phi-3-mini-4k-instruct",
    "Gemma-2B (Light)": "google/gemma-2-2b-it",
}

selected_model = st.selectbox("Choose model:", list(MODELS.keys()))
model_id = MODELS[selected_model]

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Correct API URL format
API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def chat_with_hf(messages):
    """Send chat to Hugging Face API"""
    
    # Format conversation for the model
    formatted_prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            formatted_prompt += f"User: {msg['content']}\n"
        else:
            formatted_prompt += f"Assistant: {msg['content']}\n"
    formatted_prompt += "Assistant:"
    
    # Prepare request
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False
        }
    }
    
    try:
        # Make POST request
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            # Handle different response formats
            if isinstance(result, list):
                return result[0].get("generated_text", str(result))
            elif isinstance(result, dict):
                return result.get("generated_text", str(result))
            else:
                return str(result)
        else:
            return f"Error {response.status_code}: {response.text[:200]}"
            
    except Exception as e:
        return f"Exception: {str(e)}"

# Chat input
if prompt := st.chat_input("Say something..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_with_hf(st.session_state.messages)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
