import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
import numpy as np

# Load dataset
with open("datasetDL.json", "r") as file:
    data = json.load(file)["intents"]

# Flatten dataset: pertanyaan dan respons
questions = []
responses = []
for intent in data:
    questions.extend(intent["text"])
    responses.extend(intent["responses"])

# Load pretrained model
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

# Encode all questions
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Generate chatbot response
def generate_response(user_input, threshold=0.7):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, question_embeddings)
    max_idx = similarities.argmax().item()
    max_similarity = similarities[0, max_idx].item()

    if max_similarity > threshold:
        return responses[max_idx]
    else:
        return "Maaf, saya tidak memahami pertanyaan Anda. Bisa Anda jelaskan lebih lanjut?"

# Streamlit interface
st.title("Chatbot Tentang HIV")
st.write("Halo! Saya di sini untuk membantu Anda dengan pertanyaan terkait HIV. 😊")

# User input
user_input = st.text_input("Masukkan pertanyaan Anda:")

if user_input:
    response = generate_response(user_input)
    st.write(f"**Bot:** {response}")
