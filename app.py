from flask import Flask, render_template, request, jsonify
import csv
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import os

app = Flask(__name__)
nlp = spacy.load('hr_core_news_sm')

def load_faq_from_csv(file_path):
    questions = []
    answers = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            questions.append(row['Pitanje'])
            answers.append(row['Odgovor'])
    return questions, answers

def preprocess_text(text):
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def find_best_match(user_input, questions):
    processed_input = preprocess_text(user_input)
    processed_questions = [preprocess_text(q) for q in questions]
    
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(processed_questions)
    user_vector = vectorizer.transform([processed_input])
    
    similarities = cosine_similarity(user_vector, question_vectors)
    best_match_index = similarities.argmax()
    
    return questions[best_match_index], similarities[0][best_match_index]

def chatbot_response(user_input, questions, answers):
    best_question, similarity = find_best_match(user_input, questions)
    if similarity > 0.5:
        answer_index = questions.index(best_question)
        return answers[answer_index]
    else:
        return "Na to pitanje nemam odgovor."

def speak_response(text):
    audio_file = "static/response.mp3"
    
    # ako postoji, izbriši je
    if os.path.exists(audio_file):
        os.remove(audio_file)

    tts = gTTS(text=text, lang='hr')
    tts.save(audio_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['question']
    questions, answers = load_faq_from_csv("lipik_qna.csv")
    response = chatbot_response(user_input, questions, answers)
    speak_response(response)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)

# python app.py