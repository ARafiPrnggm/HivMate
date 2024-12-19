import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os

# Load dataset
file_path = os.path.join(os.path.dirname(__file__), "datasetDL.json")
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Initialize model
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

# Prepare questions and responses
questions = []
responses = []
for intent in data["intents"]:
    questions.extend(intent["text"])
    responses.extend(intent["responses"])

question_embeddings = model.encode(questions)

# Configure Streamlit page
st.set_page_config(page_title="Chatbot Edukasi HIV", page_icon="🎗️", layout="wide")

# Add logos at the top
st.markdown(
    """
    <div style="text-align: center; display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
        <img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.itera.ac.id%2Fpengumuan-ujian-seleksi-mandiri-usm-itera%2Flogo-itera-oke%2F&psig=AOvVaw0BwqoyopUDh0m3YMpw4Z7Z&ust=1734657703016000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCNiCn9PVsooDFQAAAAAdAAAAABAE"  alt="Logo 3" width="70"/>
        <img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Ftwitter.com%2Fsainsdataitera&psig=AOvVaw1FvoT0EezzasbwwwQKTfA1&ust=1734657601186000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCODN-bHVsooDFQAAAAAdAAAAABAE" alt="Logo 2" width="70"/>
        <img src="/deploy/hivmate.png" alt="Logo 3" width="70"/>
    </div>
    """,
    unsafe_allow_html=True
)

# Add custom styling
st.markdown(
    """
    <style>
        .title {
            color: #d90429;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #333;
            font-size: 18px;
            text-align: center;
            margin-bottom: 20px;
        }
        .chat-container {
            background-color: #eaf0f6;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 700px;
            margin: auto;
            overflow-y: auto;
            max-height: 500px;
            border: 1px solid #d9d9d9;
        }
        .chat-bubble-user {
            background-color: #0088cc;
            color: #ffffff;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            max-width: 70%;
            float: left;
            clear: both;
        }
        .chat-bubble-bot {
            background-color: #5a5a5a;
            color: #ffffff;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            max-width: 70%;
            float: right;
            clear: both;
        }
        .response {
            font-size: 16px;
        }
        .send-button {
            background-color: #0088cc;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 20px;
            cursor: pointer;
            width: 100%;
        }
        .send-button:hover {
            background-color: #005f99;
        }
    </style>
    <div class="title">Chatbot Edukasi HIV 🎗️</div>
    <div class="subtitle">Temukan informasi terpercaya seputar HIV/AIDS di sini!</div>
    """,
    unsafe_allow_html=True
)

# Chat container
st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.text_input("Tulis pertanyaan Anda di sini:", placeholder="Contoh: Apa itu HIV?", key="user_input")

if user_input:
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    best_match_idx = np.argmax(similarities)
    response = responses[best_match_idx]
    st.session_state["chat_history"].append({"user": user_input, "bot": response})

# Display chat history
for chat in st.session_state["chat_history"]:
    st.markdown(f'<div class="chat-bubble-user">{chat["user"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-bubble-bot">{chat["bot"]}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; font-size: 12px; color: #888;">
        Dibuat oleh <strong>[Nama Anda]</strong>. Chatbot ini bertujuan untuk meningkatkan kesadaran tentang HIV/AIDS. 
        Jika membutuhkan informasi lebih lanjut, silakan konsultasi dengan profesional medis.
    </div>
    """,
    unsafe_allow_html=True
)

if st.button("Kirim", key="send_button"):
    user_input = st.session_state.get("user_input", "")
    if user_input:
        st.text_input("Tulis pertanyaan Anda di sini:", placeholder="Contoh: Apa itu HIV?", key="user_input", value="")
