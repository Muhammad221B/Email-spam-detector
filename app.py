import streamlit as st
import joblib
import sys
from pathlib import Path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.preprocessing import preprocess_text

rf_model = joblib.load("models/rf_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

st.title("📩 Spam Detector")

user_input = st.text_area("✉️ Enter the email message:")

if st.button("🔍 Message classification"):
    if user_input.strip() == "":
        st.warning("ًenter a message first.")
    else:
        cleaned_input = preprocess_text(user_input)
        vectorized_input = tfidf.transform([cleaned_input]).toarray()
        prediction = rf_model.predict(vectorized_input)

        if prediction[0] == 1:
            st.error("📛 Spam")
        else:
            st.success("✅ Ham")
