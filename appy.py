import streamlit as st
from huggingface_hub import InferenceClient

# 1. Page Configuration
st.set_page_config(page_title="Aura AI", page_icon="🌐")

# 2. Custom CSS - Fixed the unsafe_allow_html parameter
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
    </style>
    """, unsafe_allow_html=True)

st.title("🌐 Aura AI")
st.caption("Next-Gen Intelligent Assistant | Powered by Gemma-2")

# 3. Secure Token Access
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    st.error("Credential Error: Please add your HF_TOKEN to Streamlit Secrets.")
    st.stop()

# 4. Initialize Model - Switched to 9b for better serverless support
client = InferenceClient(model="google/gemma-2-9b-it", token=HF_TOKEN)

# 5. Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Interaction Logic - Using chat_completion for 'conversational' task
if prompt := st.chat_input("Illuminate your thoughts..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response_placeholder = st.empty()
            full_response = ""
            
            # Streaming the response for a fancy real-time feel
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
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Aura Connection Interrupted: {e}")
