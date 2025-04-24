import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import docx
from transformers import pipeline
import json

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model for semantic similarity
similarity_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Read from .docx file
def read_word_file(file):
    doc = docx.Document(file)
    return [para.text.strip() for para in doc.paragraphs if para.text.strip()]

# Contextual similarity calculation
def calculate_contextual_similarity(answer, correct_answer):
    answer_vec = np.mean(similarity_model(answer)[0], axis=0)
    correct_vec = np.mean(similarity_model(correct_answer)[0], axis=0)
    norm_answer = answer_vec / np.linalg.norm(answer_vec)
    norm_correct = correct_vec / np.linalg.norm(correct_vec)
    similarity = np.dot(norm_answer, norm_correct)
    return similarity

# Grade answers
def grade_answers_generalized(questions, correct_answers, student_answers):
    scores = []
    for i, (question, correct_answer) in enumerate(zip(questions, correct_answers)):
        student_answer = student_answers[i] if i < len(student_answers) else ""
        similarity = calculate_contextual_similarity(student_answer, correct_answer)
        normalized_score = max(min(similarity, 1), 0)
        score = round(normalized_score * 5, 2)
        scores.append((question, correct_answer, student_answer, score))
    return scores


# ---------------- Streamlit UI ------------------

st.set_page_config(page_title="GradeSmasher", layout="centered", page_icon="📝")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📘 GradeSmasher</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Auto-grade short answers with contextual AI intelligence.</p>", unsafe_allow_html=True)
st.markdown("---")

# Upload Dataset
st.subheader("📤 Upload Question & Answer Dataset")
st.caption("Supported formats: .docx only")
uploaded_dataset = st.file_uploader("Upload Questions + Correct Answers", type=["docx"])

if uploaded_dataset:
    st.success("✅ Dataset uploaded successfully.")

# Upload Student Answers
st.subheader("🧑‍🎓 Upload Student's Answers")
uploaded_answers = st.file_uploader("Upload Student Answer Sheet", type=["docx"])

if uploaded_answers:
    st.success("✅ Student answer file uploaded successfully.")

st.markdown("---")

# Submit button
submit = st.button("🚀 Submit for Evaluation")

if submit:
    if uploaded_dataset and uploaded_answers:
        try:
            dataset_content = read_word_file(uploaded_dataset)
            answers_content = read_word_file(uploaded_answers)

            questions = dataset_content[::2]
            correct_answers = dataset_content[1::2]

            results = grade_answers_generalized(questions, correct_answers, answers_content)

            st.markdown("## 📝 Grading Results")
            st.markdown("---")

            for i, (question, correct_answer, student_answer, score) in enumerate(results):
                color = "green" if score >= 4 else "orange" if score >= 2.5 else "red"
                st.markdown(f"""
                <div style="border:1px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <h5>Q{i+1}: {question}</h5>
                    <b>✅ Correct:</b> {correct_answer}<br>
                    <b>🧠 Student:</b> {student_answer}<br>
                    <b>🎯 Score:</b> <span style='color:{color}'>{score}/5</span>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error while grading: {e}")
    else:
        st.warning("⚠️ Please upload both the dataset and student answer file.")
