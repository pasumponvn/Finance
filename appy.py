import streamlit as st
from huggingface_hub import InferenceClient

# 1. Page Configuration
st.set_page_config(page_title="Aura AI", page_icon="🌐")

# 2. Custom CSS (Corrected to avoid TypeError)
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stTitle {
        background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌐 Aura AI")

# 3. Model Selection
MODELS = {
    "Gemma-2B (Light)": "google/gemma-2-2b-it",
    "Mistral-7B (Best)": "mistralai/Mistral-7B-Instruct-v0.3",
    "Phi-3-Mini (Fast)": "microsoft/Phi-3-mini-4k-instruct",
}
selected_model = st.selectbox("Choose Aura's Core:", list(MODELS.keys()))
model_id = MODELS[selected_model]

# 4. Token & Client Setup
HF_TOKEN = st.secrets.get("HF_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = st.sidebar.text_input("Hugging Face Token:", type="password")
    if not HF_TOKEN:
        st.info("Please provide a token to begin.")
        st.stop()

# Initialize the official client
client = InferenceClient(api_key=HF_TOKEN)

# 5. Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Chat Logic
if prompt := st.chat_input("Illuminate your thoughts..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # chat_completion is much more stable than raw requests.post
            for chunk in client.chat_completion(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7,
                stream=True,
            ):
                token = chunk.choices[0].delta.content
                if token:
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Aura Connection Interrupted: {str(e)}")
