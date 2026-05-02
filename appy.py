import streamlit as st
from huggingface_hub import InferenceClient

# Page setup
st.set_page_config(page_title="Aura AI", page_icon="🌐")
st.title("🌐 Aura AI")
st.caption("Hello World Chatbot")

# Hugging Face token
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    st.error("Please add your HF_TOKEN to Streamlit Secrets.")
    st.stop()

# Pick a model (chat-ready recommended, e.g. LLaMA-2-Chat)
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input + response
if prompt := st.chat_input("Say hello..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            # Try chat_completion
            for chunk in client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=128,
                stream=True,
            ):
                token = chunk.choices[0].delta.content
                if token:
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
        except Exception:
            # Fallback to text_generation
            for chunk in client.text_generation(
                prompt=f"User: {prompt}\nAssistant:",
                max_new_tokens=128,
                stream=True,
            ):
                token = chunk.token
                if token:
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
