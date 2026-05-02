import streamlit as st
from huggingface_hub import InferenceClient

# 1. Page setup
st.set_page_config(page_title="HF ChatBot", page_icon="💬")
st.title("💬 Hugging Face ChatBot")

# 2. Get HF Token
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = st.text_input("Enter your Hugging Face token:", type="password")
    if not HF_TOKEN:
        st.info("Get your token from huggingface.co/settings/tokens")
        st.stop()

# 3. WORKING MODELS for chat (tested and confirmed)
MODEL_OPTIONS = {
    "Gemma-2-2B (Fast)": "google/gemma-2-2b-it",
    "Phi-3-mini (Good)": "microsoft/Phi-3-mini-4k-instruct",
    "Llama-3.2-1B (Very Fast)": "meta-llama/Llama-3.2-1B-Instruct",
    "Mistral-7B (Best Quality)": "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen2.5-1.5B (Balanced)": "Qwen/Qwen2.5-1.5B-Instruct",
    "DeepSeek-Coder-1.3B (Code)": "deepseek-ai/deepseek-coder-1.3b-instruct",
}

st.sidebar.title("⚙️ Settings")
selected_model_name = st.sidebar.selectbox(
    "Choose model:",
    options=list(MODEL_OPTIONS.keys()),
    index=0
)
selected_model = MODEL_OPTIONS[selected_model_name]

# Advanced settings
temperature = st.sidebar.slider("Temperature:", 0.1, 1.5, 0.7, 0.1)
max_tokens = st.sidebar.slider("Max tokens:", 50, 1000, 256, 50)

st.sidebar.info(f"""
**Active Model:** {selected_model_name}
**Temperature:** {temperature}
**Max Tokens:** {max_tokens}

💡 Tip: Lower temperature = more focused responses
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

# 5. Generate response function
def get_chat_response(messages, model, temp, max_tok):
    """Get response from Hugging Face API"""
    try:
        # Prepare messages in correct format
        chat_messages = []
        for msg in messages:
            chat_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Call Hugging Face chat completion
        response = client.chat_completion(
            model=model,
            messages=chat_messages,
            temperature=temp,
            max_tokens=max_tok,
            stream=False
        )
        
        return response["choices"][0]["message"]["content"]
    
    except Exception as e:
        # Try alternative method for some models
        try:
            # Build prompt manually
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                else:
                    prompt += f"Assistant: {msg['content']}\n"
            prompt += "Assistant:"
            
            # Use text generation as fallback
            response = client.text_generation(
                prompt=prompt,
                model=model,
                temperature=temp,
                max_new_tokens=max_tok,
            )
            return response
        
        except Exception as e2:
            return f"❌ Error: {str(e2)[:200]}"

# 6. Chat input and response
if prompt := st.chat_input("Type your message here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            response = get_chat_response(
                st.session_state.messages,
                selected_model,
                temperature,
                max_tokens
            )
            st.markdown(response)
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

# 7. Sidebar buttons
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("📊 Stats"):
        msg_count = len(st.session_state.messages)
        st.sidebar.info(f"Messages: {msg_count}\nUsers: {msg_count//2} exchanges")

# 8. Example prompts
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Examples")
if st.sidebar.button("👋 Say Hello"):
    st.session_state.messages.append({"role": "user", "content": "Hello! How are you?"})
    st.rerun()

if st.sidebar.button("💡 Tell me a joke"):
    st.session_state.messages.append({"role": "user", "content": "Tell me a short joke"})
    st.rerun()

if st.sidebar.button("📝 Explain AI"):
    st.session_state.messages.append({"role": "user", "content": "What is artificial intelligence in simple terms?"})
    st.rerun()
