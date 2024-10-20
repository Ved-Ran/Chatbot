from flask import Flask, render_template, request, jsonify
import csv
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import os

app = Flask(__name__)
nlp = spacy.load('hr_core_news_sm')  # Load the Croatian language model using spaCy

# Function to load questions and answers from a CSV file
def load_faq_from_csv(file_path):
    questions = []
    answers = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            questions.append(row['Pitanje'])  # Read each question
            answers.append(row['Odgovor'])  # Read the corresponding answer
    return questions, answers

# Function to preprocess the text using lemmatization and removing stop words and punctuation
def preprocess_text(text):
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Function to find the best matching question for the user's input using cosine similarity
def find_best_match(user_input, questions):
    processed_input = preprocess_text(user_input)
    processed_questions = [preprocess_text(q) for q in questions]
    
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(processed_questions)  # Create TF-IDF vectors for questions
    user_vector = vectorizer.transform([processed_input])  # Transform the user's input into a TF-IDF vector
    
    similarities = cosine_similarity(user_vector, question_vectors)  # Calculate cosine similarity between user input and questions
    best_match_index = similarities.argmax()  # Get the index of the best matching question
    
    return questions[best_match_index], similarities[0][best_match_index]

# Function to generate the chatbot's response based on the user's input
def chatbot_response(user_input, questions, answers):
    best_question, similarity = find_best_match(user_input, questions)
    if similarity > 0.5:  # If the similarity is greater than 0.5, consider it a good match
        answer_index = questions.index(best_question)
        return answers[answer_index]  # Return the answer to the best matching question
    else:
        return "Na to pitanje nemam odgovor."  # Return a default response if no good match is found

# Function to convert text to speech using Google's Text-to-Speech (gTTS)
def speak_response(text):
    audio_file = "static/response.mp3"
    
    # If the audio file already exists, delete it to create a new one
    if os.path.exists(audio_file):
        os.remove(audio_file)

    tts = gTTS(text=text, lang='hr')  # Generate the audio in Croatian
    tts.save(audio_file)  # Save the audio file

# Route to render the main chatbot interface
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle user input and provide a response
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['question']  # Get the user's question from the form
    questions, answers = load_faq_from_csv("lipik_qna.csv")  # Load the FAQ data
    response = chatbot_response(user_input, questions, answers)  # Get the chatbot's response
    speak_response(response)  # Generate the audio response
    return jsonify({'response': response})  # Return the response as JSON

if __name__ == "__main__":
    app.run(debug=True)  # Run the app in debug mode
