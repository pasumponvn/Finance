import streamlit as st
from huggingface_hub import InferenceClient

# 1. Page setup
st.set_page_config(page_title="Hello World Chatbot", page_icon="💬")
st.title("💬 Hello World Chatbot")

# 2. Hugging Face token
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    st.error("Please add your HF_TOKEN to Streamlit Secrets.")
    st.stop()

# 3. Choose a model that is inference-enabled
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"  # chat-ready model
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
            # Try chat_completion with streaming
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
            try:
                # Fallback: non-streaming chat_completion
                result = client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=128,
                    stream=False,
                )
                full_response = result.choices[0].message.content
            except Exception:
                # Final fallback: text_generation
                result = client.text_generation(
                    prompt=f"User: {prompt}\nAssistant:",
                    max_new_tokens=128,
                )
                full_response = result.generated_text

        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
