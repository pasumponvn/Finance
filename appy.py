import streamlit as st
from huggingface_hub import InferenceClient

# 1. Page Configuration & Modern Aesthetics
st.set_page_config(page_title="Aura AI", page_icon="🌐")

# Custom CSS for a "Fancy" Tech look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTitle {
        background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 3rem !important;
    }
    div.stButton > button:first-child {
        background-color: #4facfe;
        color: white;
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌐 Aura AI")
st.caption("Next-Gen Intelligent Assistant | Powered by Gemma-2")

# 2. Secure Token Access
try:
    # This pulls from your Streamlit Cloud "Advanced Settings > Secrets"
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    st.error("Credential Error: Please add your HF_TOKEN to Streamlit Secrets.")
    st.stop()

# 3. Initialize Model
# google/gemma-2-2b-it is the engine for Aura AI
client = InferenceClient(model="google/gemma-2-2b-it", token=HF_TOKEN)

# 4. Session State for Chat Persistence
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Interaction Logic
if prompt := st.chat_input("Illuminate your thoughts..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # We use chat_completion instead of text_generation
            # This is compatible with the 'conversational' task requirement
            response = ""
            resp_container = st.empty()
            
            for chunk in client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are Aura AI, a sophisticated and visionary assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.7,
                stream=True,
            ):
                token = chunk.choices[0].delta.content
                if token:
                    response += token
                    resp_container.markdown(response + "▌")
            
            resp_container.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"Aura Connection Interrupted: {e}")
