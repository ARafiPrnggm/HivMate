from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Inisialisasi Flask app dan model
app = Flask(__name__)
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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.json  # Menerima data JSON dari request
    user_input = data.get("user_input", "").lower()

    # Hitung embedding input pengguna
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    best_match_idx = np.argmax(similarities)

    # Ambil respons berbasis kemiripan tertinggi
    response = responses[best_match_idx]
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)