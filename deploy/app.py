import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Inisialisasi model
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

# Baca data dari file JSON
with open("datasetDL.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Parsing intents dari JSON
questions = []
responses = []
for intent in data["intents"]:
    questions.extend(intent["text"])
    responses.extend(intent["responses"])

# Hitung embeddings dataset untuk prediksi berbasis ML
question_embeddings = model.encode(questions)

# Streamlit UI
st.title("Chatbot Berbasis Streamlit")
st.write("Masukkan pertanyaan Anda di bawah ini dan dapatkan respons dari chatbot.")

# Input pengguna
user_input = st.text_input("Pertanyaan Anda:", "")

if user_input:
    # Hitung embedding input pengguna
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    best_match_idx = np.argmax(similarities)

    # Ambil respons berbasis kemiripan tertinggi
    response = responses[best_match_idx]
    
    # Tampilkan hasil
    st.write("### Respons Chatbot:")
    st.write(response)
