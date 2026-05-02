import streamlit as st
from huggingface_hub import InferenceClient

# 1. Page setup
st.set_page_config(page_title="HF ChatBot", page_icon="💬")
st.title("💬 Hugging Face ChatBot - Working Version")

# 2. Get HF Token
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = st.text_input("Enter your Hugging Face token:", type="password")
    if not HF_TOKEN:
        st.info("Get your token from huggingface.co/settings/tokens")
        st.stop()

# 3. Models that work with Featherless conversational API
MODEL_OPTIONS = {
    "Gemma-2-2B (Fast)": "google/gemma-2-2b-it",
    "Phi-3-mini (Good)": "microsoft/Phi-3-mini-4k-instruct",
    "Llama-3.2-1B (Very Fast)": "meta-llama/Llama-3.2-1B-Instruct",
    "Mistral-7B (Best Quality)": "mistralai/Mistral-7B-Instruct-v0.3",
}

st.sidebar.title("⚙️ Settings")
selected_model_name = st.sidebar.selectbox(
    "Choose model:",
    options=list(MODEL_OPTIONS.keys()),
    index=0
)
selected_model = MODEL_OPTIONS[selected_model_name]

temperature = st.sidebar.slider("Temperature:", 0.1, 1.5, 0.7, 0.1)
max_tokens = st.sidebar.slider("Max tokens:", 50, 1000, 256, 50)

st.sidebar.info(f"""
**Active Model:** {selected_model_name}
**Temperature:** {temperature}
**Max Tokens:** {max_tokens}
""")

# Initialize client
client = InferenceClient(token=HF_TOKEN)

# 4. Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Function to get response using the CORRECT API
def get_response(messages, model, temp, max_tok):
    """Get response using conversational API"""
    try:
        # Format messages for conversational API
        # Convert to the format expected by the model
        conversation = ""
        for msg in messages:
            if msg["role"] == "user":
                conversation += f"User: {msg['content']}\n"
            else:
                conversation += f"Assistant: {msg['content']}\n"
        conversation += "Assistant:"
        
        # Use text_generation but with the model explicitly
        response = client.text_generation(
            prompt=conversation,
            model=model,
            temperature=temp,
            max_new_tokens=max_tok,
            stream=False
        )
        
        return response
        
    except Exception as e:
        return f"Error: {str(e)[:200]}"

# 6. Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            response = get_response(
                st.session_state.messages,
                selected_model,
                temperature,
                max_tokens
            )
            st.markdown(response)
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

# 7. Clear button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()
