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

# Set page configuration (This should be the first command)
st.set_page_config(page_title="Chatbot Edukasi HIV", page_icon="🎗️", layout="wide")

# Inisialisasi model
@st.cache_resource
def load_model():
    return SentenceTransformer("distiluse-base-multilingual-cased-v2")

model = load_model()

# Parsing intents dari JSON
questions = []
responses = []
for intent in data["intents"]:
    questions.extend(intent["text"])
    responses.extend(intent["responses"])

# Hitung embeddings dataset untuk prediksi berbasis ML
question_embeddings = model.encode(questions)

# Menambahkan CSS dengan st.markdown
st.markdown("""
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        .chat-container {
            width: 400px;
            background-color: white;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
            overflow: hidden;
        }

        h1 {
            background-color: #128cb5;
            color: white;
            margin: 0;
            padding: 20px;
            text-align: center;
        }

        #chat-box {
            height: 300px;
            padding: 20px;
            overflow-y: auto;
            background-color: #f1f1f1;
        }

        .bot-message {
            background-color: #e0ffe0;
            color: #333;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            max-width: 80%;
        }

        .user-message {
            background-color: #cce7ff;
            color: #333;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            max-width: 80%;
            align-self: flex-end;
        }

        form {
            display: flex;
            border-top: 1px solid #ddd;
        }

        input {
            flex: 1;
            padding: 10px;
            border: none;
            border-top-left-radius: 10px;
            border-bottom-left-radius: 10px;
        }

        button {
            padding: 10px;
            background-color: #128cb5;
            color: white;
            border: none;
            cursor: pointer;
            border-top-right-radius: 10px;
            border-bottom-right-radius: 10px;
        }

        button:hover {
            background-color: #128cb5;
        }
    </style>
""", unsafe_allow_html=True)

# Input pengguna
user_input = st.text_input("Tulis pertanyaan Anda di sini:", placeholder="Contoh: Apa itu HIV?", key="user_input")

# Pastikan 'chat_history' ada di session_state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if user_input.strip():
    # Hitung embedding input pengguna
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    best_match_idx = np.argmax(similarities)

    # Ambil respons berbasis kemiripan tertinggi
    response = responses[best_match_idx]

    # Simpan percakapan ke riwayat
    st.session_state["chat_history"].append({"user": user_input, "bot": response})

# Tampilkan riwayat percakapan
for chat in st.session_state["chat_history"]:
    st.write(f"**Anda:** {chat['user']}")
    st.markdown(f'<div class="response">**Chatbot:** {chat["bot"]}</div>', unsafe_allow_html=True)

# Footer dengan kredit
st.markdown("""
    <div style="text-align: center; margin-top: 50px; font-size: 12px; color: #888;">
        Dibuat oleh <strong>[Nama Anda]</strong>. Chatbot ini bertujuan untuk meningkatkan kesadaran tentang HIV/AIDS. 
        Jika membutuhkan informasi lebih lanjut, silakan konsultasi dengan profesional medis.
    </div>
""", unsafe_allow_html=True)
