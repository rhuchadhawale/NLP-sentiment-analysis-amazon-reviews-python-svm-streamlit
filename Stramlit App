import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# --- Load Models and Vectorizers ---
@st.cache_resource
def load_resources():
    # 3-sentiment model
    tfidf_vectorizer_3 = joblib.load('tfidf_vectorizer_3_sentiment.joblib')
    svm_model_3 = joblib.load('svm_model_3_sentiment.joblib')
    
    # 2-sentiment model
    tfidf_vectorizer_2 = joblib.load('tfidf_vectorizer_2_sentiment.joblib')
    svm_model_2 = joblib.load('svm_model_2_sentiment.joblib')
    
    # Initialize NLTK components
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    return tfidf_vectorizer_3, svm_model_3, tfidf_vectorizer_2, svm_model_2, stop_words, lemmatizer

tfidf_vectorizer_3, svm_model_3, tfidf_vectorizer_2, svm_model_2, stop_words, lemmatizer = load_resources()

# --- Text Preprocessing Function (from notebook) ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Remove URLs (http/https)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Remove non-ASCII characters (already handled by emoji.demojize in original, but ensure clean here)
    # Keeping only letters and spaces after removing special characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 4. Remove extra whitespace/newlines
    text = " ".join(text.split())
    
    # 5. Word Tokenization
    tokens = word_tokenize(text)
    
    # 6. Stop words removal
    tokens_clean = [word for word in tokens if word not in stop_words]
    
    # 7. Lemmatization
    tokens_lemmatized = [lemmatizer.lemmatize(word) for word in tokens_clean]
    
    return ' '.join(tokens_lemmatized)

# --- Streamlit App UI ---
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("Customer Review Sentiment Analysis")
st.write("Enter a customer review below to get its sentiment prediction.")

review_input = st.text_area("Enter Review Here:", height=150)

if st.button("Analyze Sentiment"):
    if review_input:
        processed_review = preprocess_text(review_input)
        
        # Predict with 3-sentiment model
        X_tfidf_3 = tfidf_vectorizer_3.transform([processed_review])
        sentiment_3 = svm_model_3.predict(X_tfidf_3)[0]
        
        # Predict with 2-sentiment model
        X_tfidf_2 = tfidf_vectorizer_2.transform([processed_review])
        sentiment_2 = svm_model_2.predict(X_tfidf_2)[0]
        
        st.subheader("Prediction Results:")
        st.write(f"**3-Sentiment Model (Negative, Neutral, Positive):** {sentiment_3}")
        st.write(f"**2-Sentiment Model (Negative, Positive):** {sentiment_2}")
        
    else:
        st.warning("Please enter a review to analyze.")
