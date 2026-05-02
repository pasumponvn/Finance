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
    "DeepSeek V4": "deepseek-ai/DeepSeek-V4-Flash"  # Free via HF Inference API
}

# Add model selector in sidebar
st.sidebar.title("Model Settings")
selected_model_name = st.sidebar.selectbox(
    "Choose a model:",
    options=list(MODEL_OPTIONS.keys()),
    index=0
)

selected_model = MODEL_OPTIONS[selected_model_name]

# Add temperature control
temperature = st.sidebar.slider(
    "Temperature (creativity):",
    min_value=0.1,
    max_value=1.5,
    value=0.7,
    step=0.1
)

# Add max tokens control
max_tokens = st.sidebar.number_input(
    "Max response length (tokens):",
    min_value=50,
    max_value=2048,
    value=512,
    step=64
)

# Display current model info
st.sidebar.info(f"**Active Model:** {selected_model_name}\n\n**Model ID:** `{selected_model}`\n\n*Each model is accessed for free via Hugging Face's Inference API*")

# Initialize client
client = InferenceClient(model=selected_model, token=HF_TOKEN)

# 4. Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Input + response with conversation context
if prompt := st.chat_input("Say something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Build conversation context specific to model format
        if selected_model_name == "Mistral-7B-Instruct":
            # Mistral uses special [INST] format
            conversation = "<s>"
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    conversation += f"[INST] {msg['content']} [/INST] "
                else:
                    conversation += f"{msg['content']} </s> "
            conversation = conversation.strip()
        else:
            # Generic format for other models
            conversation = ""
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    conversation += f"User: {msg['content']}\n"
                else:
                    conversation += f"Assistant: {msg['content']}\n"
            conversation += "Assistant:"

        try:
            # Text generation with streaming
            stream = client.text_generation(
                prompt=conversation,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            
            # Process streaming response
            for chunk in stream:
                if chunk.token:
                    full_response += chunk.token
                    response_placeholder.markdown(full_response + "▌")
                    
        except Exception as e:
            # Fallback: non-streaming call
            st.warning(f"Streaming failed, using non-streaming mode: {str(e)[:100]}")
            try:
                result = client.text_generation(
                    prompt=conversation,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                )
                full_response = result
            except Exception as e2:
                full_response = f"Error: Could not generate response. {str(e2)}"

        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
