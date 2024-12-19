import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os

file_path = os.path.join(os.path.dirname(__file__), "datasetDL.json")
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Inisialisasi model
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

questions = []
responses = []
for intent in data["intents"]:
    questions.extend(intent["text"])
    responses.extend(intent["responses"])

question_embeddings = model.encode(questions)

# Set up Streamlit page
st.set_page_config(page_title="Chatbot Edukasi HIV", page_icon="🎗️", layout="wide")

# Add header styling
st.markdown("""
    <style>
        .header {
            background-color: #336699;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .header h1 {
            color: #ffffff;
            font-size: 36px;
            margin: 0;
        }
        .header p {
            color: #cccccc;
            font-size: 16px;
            margin: 5px 0 0;
        }
    </style>
    <div class="header">
        <h1>🎗️ Chatbot Edukasi HIV</h1>
        <p>Temukan informasi terpercaya seputar HIV/AIDS di sini!</p>
    </div>
""", unsafe_allow_html=True)

# Add custom chat container styling
st.markdown("""
    <style>
        .chat-container {
            background-color: #f0f8ff;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            max-width: 750px;
            margin: auto;
            overflow-y: auto;
            max-height: 600px;
        }
        .chat-bubble-user {
            background-color: #008cba;
            color: white;
            padding: 10px;
            border-radius: 15px;
            margin: 10px 0;
            max-width: 75%;
            float: left;
            clear: both;
        }
        .chat-bubble-bot {
            background-color: #333;
            color: white;
            padding: 10px;
            border-radius: 15px;
            margin: 10px 0;
            max-width: 75%;
            float: right;
            clear: both;
        }
        .response {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Chat history container
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

for chat in st.session_state["chat_history"]:
    st.markdown(f'<div class="chat-bubble-user">{chat["user"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-bubble-bot">{chat["bot"]}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer with attribution
st.markdown("""
    <div style="text-align: center; margin-top: 30px; font-size: 14px; color: #555;">
        Dibuat oleh <strong>A Rafi Paringgom Iwari</strong>. Chatbot ini bertujuan untuk meningkatkan kesadaran tentang HIV/AIDS. 
        Jika membutuhkan informasi lebih lanjut, silakan konsultasi dengan profesional medis.
    </div>
""", unsafe_allow_html=True)

if st.button("Kirim", key="send_button"):
    user_input = st.session_state.get("user_input", "")
    if user_input:
        st.text_input("Tulis pertanyaan Anda di sini:", placeholder="Contoh: Apa itu HIV?", key="user_input", value="")
