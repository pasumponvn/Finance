import streamlit as st
from huggingface_hub import InferenceClient

# 1. Page setup
st.set_page_config(page_title="Hello World TextGen", page_icon="💬")
st.title("💬 Hello World Text Generation Bot")

# 2. Hugging Face token
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    st.error("Please add your HF_TOKEN to Streamlit Secrets.")
    st.stop()

# 3. Choose a model that supports text-generation
MODEL_ID = "tiiuae/falcon-7b-instruct"   # example instruct model
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# 4. Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Input + response
if prompt := st.chat_input("Say something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            # Text generation with streaming
            for chunk in client.text_generation(
                prompt=f"User: {prompt}\nAssistant:",
                max_new_tokens=128,
                temperature=0.7,
                stream=True,
            ):
                token = chunk.token
                if token:
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
        except Exception as e:
            # Fallback: non-streaming call
            result = client.text_generation(
                prompt=f"User: {prompt}\nAssistant:",
                max_new_tokens=128,
                temperature=0.7,
                stream=False,
            )
            full_response = result.generated_text

        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
