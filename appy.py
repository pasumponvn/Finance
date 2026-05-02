import streamlit as st
from huggingface_hub import InferenceClient

# 1. Page setup
st.set_page_config(page_title="Free ChatBot", page_icon="💬")
st.title("💬 Free ChatBot - Working Models")

# 2. Hugging Face token
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    st.error("Please add your HF_TOKEN to Streamlit Secrets.")
    st.stop()

# 3. CONFIRMED WORKING FREE MODELS
MODEL_OPTIONS = {
    "Mistral-7B (Fast)": "mistralai/Mistral-7B-Instruct-v0.1",
    "Gemma-2B (Lightweight)": "google/gemma-2b-it",
    "Llama-3.2-1B (Very Fast)": "meta-llama/Llama-3.2-1B-Instruct",
    "Phi-3-mini (Good Quality)": "microsoft/Phi-3-mini-4k-instruct",
    "SmolLM-1.7B (Tiny)": "HuggingFaceTB/SmolLM-1.7B-Instruct",
    "DeepSeek-Coder (Code Focus)": "deepseek-ai/deepseek-coder-1.3b-instruct",
}

st.sidebar.title("Model Settings")
selected_model_name = st.sidebar.selectbox(
    "Choose a model:",
    options=list(MODEL_OPTIONS.keys()),
    index=0
)

selected_model = MODEL_OPTIONS[selected_model_name]

# Controls
temperature = st.sidebar.slider(
    "Temperature:",
    min_value=0.1,
    max_value=1.5,
    value=0.7,
    step=0.1
)

max_tokens = st.sidebar.number_input(
    "Max tokens:",
    min_value=50,
    max_value=1024,
    value=256,
    step=32
)

st.sidebar.info(f"**Model:** {selected_model_name}\n\n*Free via Hugging Face Inference API*")

# Initialize client
client = InferenceClient(token=HF_TOKEN)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle input
if prompt := st.chat_input("Say something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Prepare messages
        api_messages = []
        for msg in st.session_state.messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
        
        try:
            # Try chat completion first
            response = client.chat.completions.create(
                model=selected_model,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,  # Use non-streaming for better compatibility
            )
            full_response = response.choices[0].message.content
            response_placeholder.markdown(full_response)
            
        except Exception as e:
            # Fallback to text generation for models that don't support chat API
            try:
                # Build prompt for text generation
                prompt_text = ""
                for msg in api_messages:
                    if msg["role"] == "user":
                        prompt_text += f"User: {msg['content']}\n"
                    else:
                        prompt_text += f"Assistant: {msg['content']}\n"
                prompt_text += "Assistant:"
                
                result = client.text_generation(
                    prompt=prompt_text,
                    model=selected_model,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                full_response = result
                response_placeholder.markdown(full_response)
                
            except Exception as e2:
                error_msg = f"⚠️ Error: {str(e2)[:200]}"
                full_response = error_msg
                response_placeholder.markdown(error_msg)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
