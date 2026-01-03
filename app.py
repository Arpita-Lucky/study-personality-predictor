# Study technique prediction app using ML model and Gemini AI
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import time
import os

# --- 1. PAGE SETUP ---
# Configure the Streamlit page with custom title, icon, and centered layout
st.set_page_config(page_title="StudyPredict AI", page_icon="🎓", layout="centered")

# Set a soft gradient background for the entire app
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #f5f7ff 0%, #e6ecff 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 2. LOAD YOUR ML BRAIN (.pkl files) ---
# Cached function to load pre-trained ML model and feature scaler
# Only runs once per session for performance optimization
@st.cache_resource
def load_ml_assets():
    try:
        # Check if both required pickle files exist in the working directory
        if os.path.exists('personality_model.pkl') and os.path.exists('scaler.pkl'):
            # Load the trained personality prediction model
            with open('personality_model.pkl', 'rb') as f:
                model = pickle.load(f)
            # Load the feature scaler (used to normalize Big Five trait inputs)
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            return model, scaler
    except Exception as e:
        st.error(f"Error loading files: {e}")
    return None, None

model, scaler = load_ml_assets()

# --- 3. GEMINI AI FOR SMART ADVICE ---
# Calls Gemini AI to generate personalized study advice and 7-day roadmap
def get_ai_advice(profile, techniques, subjects):
    # API key is provided by the environment at runtime
    api_key = "AIzaSyA9R5iU7eO2JRETy5Ig_f1Phv7EPN6sYQM" 
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    
    # Create a prompt that instructs Gemini to act as a study coach
    # Combines personality traits, predicted techniques, and subject list
    prompt = f"""
    Act as a professional Study Success Coach for Arpita Lakhisirani's project.
    Explain why these study techniques ({techniques}) fit a student with these traits ({profile}). 
    Then, create a 7-day study schedule for these subjects: {subjects}.
    Keep it professional and encouraging!
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    # Retry logic with exponential backoff (1s, 2s, 4s delays)
    for delay in [1, 2, 4]:
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                # Extract and return the generated text from Gemini's response
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            time.sleep(delay)
        except:
            continue
    # Fallback message if all retry attempts fail
    return "AI Insight currently unavailable, but your model predictions are below!"

# --- 4. THE INTERFACE ---
# Main UI displaying title and input controls
st.title("🎓 StudyPredict AI")
st.write(f"Final Project by Arpita Lakhisirani and Fahad Hucane")
 
if model is None or scaler is None:
    st.error("⚠️ Files missing! Put app.py, personality_model.pkl, and scaler.pkl in the SAME folder.")
else:
    # Text input for user's study subjects
    subjects = st.text_input(" What are you studying?", "Data Science, Python")
    
    # Big Five personality trait sliders (1-5 Likert scale)
    st.write("####  Your Personality (Rate 1-5)")
    col1, col2 = st.columns(2)
    with col1:
        o = st.slider("Openness: I enjoy thinking about complex ideas and trying new things.", 1, 5, 3)  # Curiosity, creativity
        c = st.slider("Conscientiousness: I am organized and disciplined.", 1, 5, 3)  # Organization, discipline
        e = st.slider("Extraversion: I am outgoing and energetic.", 1, 5, 3)  # Social energy
    with col2:
        a = st.slider("Agreeableness: I am empathetic and cooperative.", 1, 5, 3)  # Empathy, cooperation
        n = st.slider("Neuroticism: I am sensitive to stress.", 1, 5, 3)  # Stress sensitivity

    # Prediction button triggers ML inference and AI roadmap generation
    if st.button(" Run Prediction", type="primary"):
        # Prepare input: Big Five traits as a 2D numpy array (required for sklearn)
        input_data = np.array([[o, c, e, a, n]])
        # Normalize input using the same scaler from training
        scaled_input = scaler.transform(input_data)
        # Get binary predictions for each study technique (multi-label classification)
        prediction = model.predict(scaled_input)[0]
        
        # Map binary outputs to human-readable technique names
        tech_names = ['Spaced Repetition', 'Active Recall', 'Feynman Technique', 'Mind Mapping', 'Pomodoro Technique']
        results = [tech_names[i] for i, val in enumerate(prediction) if val == 1]
        
        st.divider()
        # Display ML model's recommended techniques (default to Active Recall if none predicted)
        st.success(f"### 🤖 Model Result: {', '.join(results) if results else 'Active Recall'}")
        
        # Call Gemini AI to generate personalized study roadmap
        with st.spinner("Gemini AI is generating your roadmap..."):
            # Format personality profile as comma-separated string for AI prompt
            summary = f"O:{o}, C:{c}, E:{e}, A:{a}, N:{n}"
            ai_text = get_ai_advice(summary, results, subjects)
            st.markdown("### ✨ AI Personal Roadmap")
            st.write(ai_text)