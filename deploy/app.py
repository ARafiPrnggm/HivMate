import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer, util

# Load dataset
with open("datasetDL.json", "r") as file:
    data = json.load(file)["intents"]

# Flatten dataset: pertanyaan dan respons
questions = []
responses = []
for intent in data:
    questions.extend(intent["text"])
    responses.extend(intent["responses"])

# Load model Sentence Transformer
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
# Menyimpan model
model.save("./model")

# Encode semua pertanyaan di dataset
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Data uji: Pertanyaan dan jawaban tentang HIV
test_data = [
    {"input": "Apa itu HIV?", "expected_response": "HIV adalah virus yang menyerang sistem kekebalan tubuh."},
    {"input": "Bagaimana cara penularan HIV?", "expected_response": "HIV dapat menular melalui kontak dengan darah, hubungan seksual, atau dari ibu ke anak selama kehamilan."},
    {"input": "Apa gejala awal HIV?", "expected_response": "Gejala awal HIV mirip dengan flu, seperti demam, sakit kepala, dan ruam kulit."},
    {"input": "Bisakah HIV disembuhkan?", "expected_response": "Saat ini belum ada obat untuk menyembuhkan HIV, tetapi pengobatan antiretroviral (ARV) dapat mengontrol virus."},
    {"input": "Bagaimana cara mencegah HIV?", "expected_response": "Menggunakan kondom, tidak berbagi jarum suntik, dan melakukan tes HIV secara rutin dapat mencegah penularan HIV."},

]

# Evaluasi
correct_predictions = 0
threshold = 0.6  # Menurunkan threshold
all_similarities = []

y_true = []
y_pred = []

for item in test_data:
    user_input = item["input"]
    expected_response = item["expected_response"]

    # Encode user input
    user_embedding = model.encode(user_input, convert_to_tensor=True)

    # Hitung kesamaan kosinus
    similarities = util.cos_sim(user_embedding, question_embeddings)
    max_idx = similarities.argmax().item()
    max_similarity = similarities[0, max_idx].item()
    all_similarities.append(max_similarity)

    predicted_response = responses[max_idx]

    print(f"Pertanyaan: {user_input}")
    print(f"Prediksi: {predicted_response}")
    print(f"Jawaban yang Diharapkan: {expected_response}")
    print(f"Similarity: {max_similarity:.2f}\n")

    if max_similarity > threshold:
        correct_predictions += 1
        y_true.append(1)
        y_pred.append(1)
    else:
        y_true.append(1)
        y_pred.append(0)

  # Menghitung akurasi
accuracy = correct_predictions / len(test_data)
print(f"Akurasi: {accuracy:.2f}")

# Menghitung rata-rata kemiripan
average_similarity = np.mean(all_similarities)
print(f"Rata-rata Similarity: {average_similarity:.2f}")

# Menghitung Precision, Recall, dan F1-Score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

def generate_response(user_input):
    """
    Generate chatbot response based on semantic similarity.
    """
    # Encode masukan pengguna
    user_embedding = model.encode(user_input, convert_to_tensor=True)

    # Encode semua pertanyaan di dataset
    question_embeddings = model.encode(questions, convert_to_tensor=True)

    # Hitung kesamaan kosinus antara masukan pengguna dan pertanyaan dataset
    similarities = util.cos_sim(user_embedding, question_embeddings)

    # Cari pertanyaan dengan kesamaan tertinggi
    max_idx = similarities.argmax().item()
    max_similarity = similarities[0, max_idx].item()

    # Threshold untuk memastikan relevansi
    if max_similarity > 0.7:  # Anda bisa sesuaikan nilai threshold ini
        return responses[max_idx]
    else:
        return "Maaf, saya tidak memahami pertanyaan Anda. Bisa Anda jelaskan lebih lanjut?"

print("Chatbot: Halo! Saya di sini untuk membantu Anda. Ketik 'keluar' untuk berhenti.")

while True:
    user_input = input("Anda: ")
    if user_input.lower() == "keluar":
        print("Chatbot: Terima kasih! Sampai jumpa.")
        break

    response = generate_response(user_input)
    print(f"Chatbot: {response}")
