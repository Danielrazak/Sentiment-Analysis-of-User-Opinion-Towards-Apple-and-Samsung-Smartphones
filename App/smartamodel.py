import joblib
import streamlit as st
from smartapreprocessing import preprocess_text
 
@st.cache_resource
def load_model():
    """Load the sentiment analysis model"""
    model = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Function to predict sentiment for new tweets with preprocessing
def predict_tweet_sentiment(tweet_text):
    cleaned_text = preprocess_text(tweet_text)
    tweet_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(tweet_vectorized)
    sentiment_mapping_reverse = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
    return sentiment_mapping_reverse[prediction[0]]


