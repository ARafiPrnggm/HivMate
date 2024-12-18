import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Inisialisasi model
st.title("Chatbot dengan Streamlit")
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

# Hitung embeddings dataset
question_embeddings = model.encode(questions)

# Fungsi untuk mendapatkan respons
def get_response(user_input):
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    best_match_idx = np.argmax(similarities)
    return responses[best_match_idx]

# Streamlit Input/Output
st.text_input("Masukkan pesan Anda:", key="user_input")
if st.session_state.user_input:
    response = get_response(st.session_state.user_input.lower())
    st.write(f"Bot: {response}")
