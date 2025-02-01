import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from textblob import TextBlob
from datetime import datetime

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    if len(results) > 0:
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        return return_list
    else:
        return None

def get_response(input_text):
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if input_text.lower() in pattern.lower():
                if '{time}' in intent['responses'][0]:
                    response = intent['responses'][0].format(time=datetime.now().strftime('%H:%M'))
                elif '{date}' in intent['responses'][0]:
                    response = intent['responses'][0].format(date=datetime.now().strftime('%d-%m-%Y'))
                elif '{joke}' in intent['responses'][0]:
                    response = intent['responses'][0].format(joke="Why don't scientists trust atoms? Because they make up everything!")
                elif '{news}' in intent['responses'][0]:
                    response = intent['responses'][0].format(news="Breaking news: Chatbot learns to provide the latest news!")
                else:
                    response = random.choice(intent['responses'])
                return response
    return "Sorry, I'm not sure how to respond to that."

st.title("Chatbot")
input_text = st.text_input("You:", "")

if input_text:
    st.write(f"You: {input_text}")
    
    # Sentiment Analysis
    sentiment_score = TextBlob(input_text).sentiment.polarity
    if sentiment_score > 0:
        st.write("User sentiment: Positive")
    elif sentiment_score < 0:
        st.write("User sentiment: Negative")
    else:
        st.write("User sentiment: Neutral")
    
    # Predicting intent
    ints = predict_class(input_text)
    if ints:
        chatbot_response = get_response(input_text)
    else:
        chatbot_response = "Sorry, I didn't understand that."
    
    st.write(f"Chatbot: {chatbot_response}")
