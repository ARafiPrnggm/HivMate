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

# Header dengan tiga logo
col1, col2, col3 = st.columns(3)
with col1:
    st.image("logo1.png", width=100)  # Ganti "logo1.png" dengan nama file logo pertama
with col2:
    st.image("logo2.png", width=100)  # Ganti "logo2.png" dengan nama file logo kedua
with col3:
    st.image("logo3.png", width=100)  # Ganti "logo3.png" dengan nama file logo ketiga

st.markdown("""
    <style>
        body {
            background-color: #f7f7f7;
        }
        .title {
            color: #d90429;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
        }
        .subtitle {
            color: #333;
            font-size: 18px;
            text-align: center;
            margin-bottom: 20px;
        }
        .chat-container {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 700px;
            margin: auto;
        }
        .user-input {
            margin-top: 20px;
        }
        .response {
            background-color: #d90429;
            color: #ffffff;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
        }
    </style>
    <div class="title">Chatbot Edukasi HIV 🎗️</div>
    <div class="subtitle">Temukan informasi terpercaya seputar HIV/AIDS di sini!</div>
""", unsafe_allow_html=True)

# Kontainer chat
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Input pengguna
user_input = st.text_input("Tulis pertanyaan Anda di sini:", key="user_input")

# Jika pengguna memberikan input
if user_input:
    # Hitung embedding input pengguna
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    best_match_idx = np.argmax(similarities)

    # Ambil respons berbasis kemiripan tertinggi
    response = responses[best_match_idx]

    # Tampilkan hasil
    st.markdown(f'<div class="response">{response}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer dengan kredit
st.markdown("""
    <div style="text-align: center; margin-top: 50px; font-size: 14px; color: #555;">
        Dibuat dengan ❤️ oleh <strong>[Nama Anda]</strong>. Model berbasis <i>distiluse-base-multilingual-cased-v2</i>. 
        Untuk informasi lebih lanjut, konsultasikan dengan profesional medis.
    </div>
""", unsafe_allow_html=True)
