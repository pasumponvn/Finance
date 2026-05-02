import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Hello World Chatbot", page_icon="💬")
st.title("💬 Hello World Chatbot")

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    st.error("Please add your HF_TOKEN to Streamlit Secrets.")
    st.stop()

# Use a model that is actually supported
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Say something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
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
            result = client.text_generation(
                prompt=f"User: {prompt}\nAssistant:",
                max_new_tokens=128,
            )
            full_response = result.generated_text

        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
