# Fungsi Reset Chat (General) 
def reset_chat_history():
    st.session_state.chat_history = []
    st.session_state.input_key = 0 

# Setup bar di dalam streamlit
st.set_page_config(
    page_title="MentiChat",
    page_icon="../assets/sun.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Setup Sidebar ada apa aja
st.sidebar.title("Mental Health AI Chat")
st.sidebar.markdown("""
Welcome to the AI-powered mental health chatbot.
Feel free to ask questions or discuss your thoughts in a safe, non-judgmental space. 💙
""")
st.sidebar.image("https://plus.unsplash.com/premium_photo-1661389446461-1e22c995e48b?q=80&w=1467&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", use_container_width=True)


# Tambahkan tombol reset di sidebar dengan styling
st.sidebar.markdown("""
    <style>
        .center-button {
            display: flex;
            justify-content: center;
            margin: 1em 0;
        }
    </style>
""", unsafe_allow_html=True)

if st.sidebar.button("🔄 Reset Chat History", key="reset_button", use_container_width=True):
    reset_chat_history()
    st.rerun()

# Sidebar - Model Selection
st.sidebar.subheader("Model Selection")
model_provider = st.sidebar.selectbox(
    "Choose a model provider:",
    ["Hugging Face", "Google"],
    key="model_provider_selectbox"
)

if model_provider == "Hugging Face":
    model_name = st.sidebar.selectbox(
        "Choose a Hugging Face model:",
        ["numind/NuExtract-1.5", "facebook/blenderbot-400M-distill", "microsoft/Phi-3.5-mini-instruct"],
        key="huggingface_model_selectbox"
    )
    HF_API_KEY = API_HF_KEY
else:
    model_name = None

if model_provider == "Google":
    if not API_DB:
        st.sidebar.error("Google API key not found in .env file!")


# Chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Chat Interface
st.markdown("""
<div style="background-color:#f0f0f5; padding:10px; border-radius:10px; text-align:center;">
    <h1 style="color:#336699;">🧠 MentiChat</h1>
    <p style="color:#666;">Your AI Mental Health Companion</p>
</div>
""", unsafe_allow_html=True)
