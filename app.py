# ===== Import necessary libraries and custom preprocessing function =====
import streamlit as st
import joblib
import sys
from pathlib import Path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))  # Add utils folder to the path
from utils.preprocessing import preprocess_text  # Import custom text preprocessing function

# ===== Load the trained model and TF-IDF vectorizer =====
rf_model = joblib.load("models/rf_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# ===== Streamlit UI setup =====
st.title("📩 Spam Detector")
user_input = st.text_area("✉️ Enter the email message:")

# ===== Handle classification when button is pressed =====
if st.button("🔍 Message classification"):
    if user_input.strip() == "":
        st.warning("ًenter a message first.")  # Show warning if input is empty
    else:
        cleaned_input = preprocess_text(user_input)  # Clean the input text
        vectorized_input = tfidf.transform([cleaned_input]).toarray()  # Vectorize the input
        prediction = rf_model.predict(vectorized_input)  # Predict using the trained model

        # Display result based on prediction
        if prediction[0] == 1:
            st.error("📛 Spam")
        else:
            st.success("✅ Ham")
