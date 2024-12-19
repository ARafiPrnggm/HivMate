import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os

# Menentukan jalur file dataset
file_path = os.path.join(os.path.dirname(__file__), "datasetDL.json")
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Inisialisasi model
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

# Parsing intents dari JSON
questions = []
responses = []
for intent in data["intents"]:
    questions.extend(intent["text"])
    responses.extend(intent["responses"])

# Hitung embeddings dataset untuk prediksi berbasis ML
question_embeddings = model.encode(questions)

# Streamlit UI dengan tema edukasi HIV
st.set_page_config(page_title="Chatbot Edukasi HIV", page_icon="🎗️", layout="wide")

# Header dengan logo
st.image("https://pbs.twimg.com/profile_images/1272461269136576512/Uw9AShxq_400x400.jpg", width=100)  # Ganti "logo.png" dengan jalur logo Anda
st.markdown("""
    <style>
        .title {
            color: #d90429;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-top: -20px;
        }
        .subtitle {
            color: #333;
            font-size: 18px;
            text-align: center;
            margin-bottom: 20px;
        }
        .chat-container {
            background-color: #f4f4f4;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 700px;
            margin: auto;
            overflow-y: auto;
            max-height: 500px;
        }
        .chat-bubble-user {
            background-color: #d90429;
            color: #ffffff;
            padding: 10px;
            border-radius: 20px;
            margin: 5px 0;
            max-width: 80%;
            float: left;
            clear: both;
            position: relative;
        }
        .chat-bubble-bot {
            background-color: #2b2d42;
            color: #ffffff;
            padding: 10px;
            border-radius: 20px;
            margin: 5px 0;
            max-width: 80%;
            float: right;
            clear: both;
            position: relative;
        }
        .response {
            font-size: 16px;
        }
        .send-button {
            background-color: #d90429;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 20px;
            cursor: pointer;
            width: 100%;
        }
        .send-button:hover {
            background-color: #b10425;
        }
    </style>
    <div class="title">Chatbot Edukasi HIV 🎗️</div>
    <div class="subtitle">Temukan informasi terpercaya seputar HIV/AIDS di sini!</div>
""", unsafe_allow_html=True)

# Kontainer chat
st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)

# Variabel untuk riwayat percakapan
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Input pengguna
user_input = st.text_input("Tulis pertanyaan Anda di sini:", placeholder="Contoh: Apa itu HIV?", key="user_input")

if user_input:
    # Hitung embedding input pengguna
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    best_match_idx = np.argmax(similarities)

    # Ambil respons berbasis kemiripan tertinggi
    response = responses[best_match_idx]

    # Simpan percakapan ke riwayat
    st.session_state["chat_history"].append({"user": user_input, "bot": response})

# Tampilkan riwayat percakapan dengan desain obrolan mirip WhatsApp
for chat in st.session_state["chat_history"]:
    st.markdown(f'<div class="chat-bubble-user">{chat["user"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-bubble-bot">{chat["bot"]}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer dengan kredit
st.markdown("""
    <div style="text-align: center; margin-top: 50px; font-size: 12px; color: #888;">
        Dibuat oleh <strong>[Nama Anda]</strong>. Chatbot ini bertujuan untuk meningkatkan kesadaran tentang HIV/AIDS. 
        Jika membutuhkan informasi lebih lanjut, silakan konsultasi dengan profesional medis.
    </div>
""", unsafe_allow_html=True)

# Tambahkan tombol kirim
if st.button("Kirim", key="send_button"):
    user_input = st.session_state.get("user_input", "")
    if user_input:
        st.text_input("Tulis pertanyaan Anda di sini:", placeholder="Contoh: Apa itu HIV?", key="user_input", value="")
