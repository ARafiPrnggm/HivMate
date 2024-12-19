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

questions = []
responses = []
for intent in data["intents"]:
    questions.extend(intent["text"])
    responses.extend(intent["responses"])

question_embeddings = model.encode(questions)

# Set Streamlit page config
st.set_page_config(page_title="Chatbot Edukasi HIV", page_icon="🎗️", layout="wide")

st.markdown(
    """
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .chat-container {
            background-color: #1E1E1E;
            padding: 20px;
            border-radius: 10px;
            max-width: 800px;
            margin: 50px auto;
            box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.5);
            font-family: Arial, sans-serif;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .header h1 {
            color: #FF3366;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .header p {
            font-size: 18px;
            color: #CCCCCC;
        }
        .logo-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
        }
        .logo-container img {
            height: 50px;
            width: auto;
        }
        .user-message {
            background-color: #FF3366;
            color: white;
            padding: 15px;
            border-radius: 15px;
            text-align: right;
            margin-bottom: 10px;
            max-width: 70%;
            float: right;
            clear: both;
            box-shadow: 0px 4px 6px rgba(255, 51, 102, 0.5);
        }
        .bot-message {
            background-color: #333333;
            color: white;
            padding: 15px;
            border-radius: 15px;
            text-align: left;
            margin-bottom: 10px;
            max-width: 70%;
            float: left;
            clear: both;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.5);
        }
        input[type=text] {
            background-color: #1E1E1E;
            color: white;
            border: 1px solid #444;
            border-radius: 5px;
            padding: 10px;
            width: 100%;
            margin-bottom: 10px;
        }
        input::placeholder {
            color: #888;
        }
        button {
            background-color: #FF3366;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #FF6699;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Header Section with logos
st.markdown(
    """
    <div class="logo-container">
        <img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fid.m.wikipedia.org%2Fwiki%2FBerkas%3ALogo_ITERA.png&psig=AOvVaw1wcIgaFB2I4cPgTUTktNpo&ust=1734661270298000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCOiax_jisooDFQAAAAAdAAAAABAR" alt="Logo 1">
        <img src="https://pbs.twimg.com/profile_images/1272461269136576512/Uw9AShxq_400x400.jpg" alt="Logo 2">
        <img src="deploy/hivmate-01.png" alt="Logo 3">
    </div>
    <div class="header">
        <h1>Chatbot Edukasi HIV 🎗️</h1>
        <p>Temukan informasi terpercaya seputar HIV/AIDS di sini!</p>
    </div>
    """,
    unsafe_allow_html=True
)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.text_input("Tulis pertanyaan Anda di sini:", placeholder="Contoh: Apa itu HIV?", key="user_input")

if user_input:
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    best_match_idx = np.argmax(similarities)
    response = responses[best_match_idx]

    st.session_state["chat_history"].append(("user", user_input))
    st.session_state["chat_history"].append(("bot", response))

for sender, message in st.session_state["chat_history"]:
    if sender == "user":
        st.markdown(f'<div class="user-message">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">{message}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
