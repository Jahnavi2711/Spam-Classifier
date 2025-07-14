import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')  # correct resource
nltk.download('stopwords')
nltk.download('punkt_tab')

# Page config
st.set_page_config(page_title="Spam Classifier", page_icon="📨", layout="centered")


st.markdown("""
    <style>
    /* Main background and text */
    [data-testid="stAppViewContainer"] {
        background-color: #121212;
        color: #e0e0e0;
    }

    [data-testid="stHeader"] {
        background: none;
    }

    h1 {
        color: #f5f5f5;
        text-align: center;
        font-size: 2.5rem;
        padding-bottom: 1rem;
    }

    .stTextArea textarea {
        background-color: #1e1e1e;
        color: #ffffff;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 12px;
        font-size: 1rem;
    }

    .stButton button {
        background-color: #3f51b5;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 10px 22px;
        margin-top: 10px;
        font-size: 1rem;
        transition: background-color 0.3s ease;
    }

    .stButton button:hover {
        background-color: #303f9f;
    }

    .result-box {
        margin-top: 25px;
        padding: 18px;
        border-radius: 10px;
        font-size: 1.3rem;
        font-weight: 500;
        text-align: center;
    }

    .not-spam {
        background-color: #2e7d32;  /* Muted green */
        color: #ffffff;
    }

    .spam {
        background-color: #c62828;  /* Muted red */
        color: #ffffff;
    }

    label, textarea {
        font-size: 1rem !important;
        color: #cccccc !important;
    }
    </style>
""", unsafe_allow_html=True)

# NLP Preprocessing
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorization.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# App Title
st.markdown("<h1>📨 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)

# Input Text
input_sms = st.text_area("📩 Enter your Email or SMS message:")

# Predict Button
if st.button("🔍 Predict"):
    # Step 1: Transform
    transformed_sms = transform_text(input_sms)

    # Step 2: Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Step 3: Predict
    result = model.predict(vector_input)[0]

    # Step 4: Output
    if result == 1:
        st.markdown('<div class="result-box spam">⚠️ This is <strong>SPAM</strong></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box not-spam">✅ This is <strong>NOT SPAM</strong></div>', unsafe_allow_html=True)
