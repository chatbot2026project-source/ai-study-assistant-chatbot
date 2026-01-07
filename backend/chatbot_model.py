import pandas as pd
import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pdf_reader import extract_text_from_pdf, chunk_text

# --------------------------------------------------
# PATH SETUP (Cloud Safe)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "study_data.csv")
PDF_PATH = os.path.join(BASE_DIR, "..", "data", "pdfs", "CYBER_SECURITY.pdf")

# --------------------------------------------------
# LOAD DATASET (CSV)
# --------------------------------------------------
data = pd.read_csv(DATA_PATH)

questions = data["question"].astype(str).values
answers = data["answer"].astype(str).values

# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

cleaned_questions = [clean_text(q) for q in questions]

# --------------------------------------------------
# TF-IDF FOR DATASET
# --------------------------------------------------
dataset_vectorizer = TfidfVectorizer()
dataset_vectors = dataset_vectorizer.fit_transform(cleaned_questions)

# --------------------------------------------------
# LOAD & PROCESS PDF
# --------------------------------------------------
pdf_text = extract_text_from_pdf(PDF_PATH)
pdf_chunks = chunk_text(pdf_text, chunk_size=120)

pdf_vectorizer = TfidfVectorizer()
pdf_vectors = pdf_vectorizer.fit_transform(pdf_chunks)

# --------------------------------------------------
# INTENT DETECTION
# --------------------------------------------------
def detect_intent(question):
    q = question.lower()
    if q.startswith("what is") or q.startswith("define"):
        return "definition"
    elif q.startswith("explain") or q.startswith("describe"):
        return "explanation"
    elif q.startswith("why"):
        return "reason"
    elif "difference" in q or "compare" in q:
        return "comparison"
    else:
        return "general"

# --------------------------------------------------
# SUBJECT DETECTION
# --------------------------------------------------
def detect_subject(question):
    q = question.lower()
    if any(word in q for word in ["deadlock", "paging", "cpu", "process", "thread"]):
        return "Operating Systems"
    elif any(word in q for word in ["dbms", "sql", "normalization", "transaction"]):
        return "DBMS"
    elif any(word in q for word in ["python", "class", "inheritance", "function"]):
        return "Programming"
    else:
        return "General Studies"

# --------------------------------------------------
# ANSWER EXPANSION (SMART PART)
# --------------------------------------------------
def expand_answer(base_answer):
    return (
        f"{base_answer}\n\n"
        f"📌 **In simple words:**\n"
        f"This concept explains how systems manage resources and operations.\n\n"
        f"🧠 **Why it is important:**\n"
        f"It helps students understand system behavior, performance, and problem solving.\n\n"
        f"📝 **Example:**\n"
        f"Consider a real-world situation where multiple users share limited resources.\n\n"
        f"🎯 **Exam Tip:**\n"
        f"This is a frequently asked topic. Write definition + one example in exams."
    )

# --------------------------------------------------
# MAIN CHATBOT FUNCTION
# --------------------------------------------------
def get_response(user_input):
    user_input_clean = clean_text(user_input)

    # ----- DATASET SIMILARITY -----
    dataset_user_vector = dataset_vectorizer.transform([user_input_clean])
    dataset_similarity = cosine_similarity(dataset_user_vector, dataset_vectors)
    ds_index = dataset_similarity.argmax()
    ds_score = dataset_similarity[0][ds_index]

    # ----- PDF SIMILARITY -----
    pdf_user_vector = pdf_vectorizer.transform([user_input_clean])
    pdf_similarity = cosine_similarity(pdf_user_vector, pdf_vectors)
    pdf_index = pdf_similarity.argmax()
    pdf_score = pdf_similarity[0][pdf_index]

    # ----- LOW CONFIDENCE CHECK -----
    if max(ds_score, pdf_score) < 0.2:
        return (
            "🤔 I’m not confident about this question.\n\n"
            "Please try rephrasing it or ask from your syllabus topics."
        )

    # ----- SELECT BEST SOURCE -----
    if pdf_score > ds_score:
        base_answer = pdf_chunks[pdf_index]
        source = "College Notes (PDF)"
    else:
        base_answer = answers[ds_index]
        source = "Prepared Dataset"

    intent = detect_intent(user_input)
    subject = detect_subject(user_input)

    # ----- RESPONSE FORMAT BASED ON INTENT -----
    if intent == "definition":
        response = f"📘 **Definition:**\n{base_answer}"

    elif intent == "explanation":
        response = expand_answer(base_answer)

    elif intent == "reason":
        response = (
            f"❓ **Reason:**\n"
            f"{base_answer}\n\n"
            f"This happens due to resource dependency and system design."
        )

    elif intent == "comparison":
        response = (
            f"🔍 **Comparison Insight:**\n"
            f"{base_answer}\n\n"
            f"I can also explain this using a comparison table."
        )

    else:
        response = expand_answer(base_answer)

    # ----- FINAL RESPONSE -----
    return (
        f"📚 **Subject:** {subject}\n"
        f"📖 **Source:** {source}\n\n"
        f"{response}\n\n"
        f"Ask me if you want this explained more simply or in exam format."
    )
