import streamlit as st
from huggingface_hub import InferenceClient

# 1. Page setup
st.set_page_config(page_title="Multi-Model ChatBot", page_icon="💬")
st.title("💬 Multi-Model ChatBot")

# 2. Hugging Face token
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    st.error("Please add your HF_TOKEN to Streamlit Secrets.")
    st.stop()

# 3. Model selection dropdown
MODEL_OPTIONS = {
    "Zephyr-7B-beta": "HuggingFaceH4/zephyr-7b-beta",
    "Phi-2": "microsoft/phi-2",
    "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "DeepSeek V4": "deepseek-ai/DeepSeek-V4-Flash"
}

st.sidebar.title("Model Settings")
selected_model_name = st.sidebar.selectbox(
    "Choose a model:",
    options=list(MODEL_OPTIONS.keys()),
    index=0
)

selected_model = MODEL_OPTIONS[selected_model_name]

# Temperature and token controls
temperature = st.sidebar.slider(
    "Temperature (creativity):",
    min_value=0.1,
    max_value=1.5,
    value=0.7,
    step=0.1
)

max_tokens = st.sidebar.number_input(
    "Max response length (tokens):",
    min_value=50,
    max_value=2048,
    value=512,
    step=64
)

st.sidebar.info(f"**Active Model:** {selected_model_name}\n\n**Model ID:** `{selected_model}`")

# Initialize client (no provider specification needed)
client = InferenceClient(token=HF_TOKEN)

# 4. Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Input + response using chat_completion
if prompt := st.chat_input("Say something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Format messages for the chat API
        # Convert session messages to the format expected by chat_completion
        api_messages = []
        for msg in st.session_state.messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

        try:
            # Use the chat_completion API instead of text_generation
            stream = client.chat.completions.create(
                model=selected_model,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            # Process streaming response
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

        except Exception as e:
            # Fallback: non-streaming chat_completion
            st.warning(f"Streaming failed, using standard mode: {str(e)[:100]}")
            try:
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=api_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                )
                full_response = response.choices[0].message.content
            except Exception as e2:
                full_response = f"Error: Could not generate response. {str(e2)}"

        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
