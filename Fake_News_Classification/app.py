import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

st.set_page_config(page_title="Fake News Classifier", page_icon="📰")
st.title("Fake News Classification 📰")
st.write("Enter news text to check whether it is Fake or Real.")

news_text = st.text_area("News Text")

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned_text = preprocess(news_text)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]

        if prediction == 1:
            st.error("Fake News ❌")
        else:
            st.success("Real News ✅")
