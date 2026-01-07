import pandas as pd
import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pdf_reader import extract_text_from_pdf, chunk_text

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "study_data.csv")
PDF_PATH = os.path.join(BASE_DIR, "..", "data", "pdfs", "Operating_System_Notes.pdf")

# ---------------- LOAD DATASET ----------------
data = pd.read_csv(DATA_PATH)
questions = data["question"].astype(str).values
answers = data["answer"].astype(str).values

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

cleaned_questions = [clean_text(q) for q in questions]

dataset_vectorizer = TfidfVectorizer()
dataset_vectors = dataset_vectorizer.fit_transform(cleaned_questions)

# ---------------- LOAD PDF SAFELY ----------------
pdf_chunks = []
pdf_vectors = None
pdf_vectorizer = None

pdf_text = extract_text_from_pdf(PDF_PATH)
if pdf_text:
    pdf_chunks = chunk_text(pdf_text)
    if pdf_chunks:
        pdf_vectorizer = TfidfVectorizer()
        pdf_vectors = pdf_vectorizer.fit_transform(pdf_chunks)

# ---------------- INTENT & SUBJECT ----------------
def detect_intent(q):
    q = q.lower()
    if q.startswith("what is") or q.startswith("define"):
        return "definition"
    if q.startswith("explain") or q.startswith("describe"):
        return "explanation"
    if q.startswith("why"):
        return "reason"
    if "difference" in q or "compare" in q:
        return "comparison"
    return "general"


def detect_subject(q):
    q = q.lower()

    if any(w in q for w in [
        "cyber", "security", "attack", "malware", "virus",
        "hacking", "phishing", "firewall", "encryption"
    ]):
        return "Cyber Security"

    if any(w in q for w in [
        "deadlock", "paging", "cpu", "process", "thread"
    ]):
        return "Operating Systems"

    if any(w in q for w in [
        "dbms", "sql", "normalization", "transaction"
    ]):
        return "DBMS"

    if any(w in q for w in [
        "python", "class", "inheritance", "function"
    ]):
        return "Programming"

    return "General Studies"


# ---------------- ANSWER EXPANSION ----------------
def expand_answer(text):
    return (
        f"{text}\n\n"
        f"📌 **In simple terms:**\n"
        f"This topic explains an important system concept.\n\n"
        f"🧠 **Why it matters:**\n"
        f"It helps in understanding how real systems work.\n\n"
        f"🎯 **Exam Tip:**\n"
        f"Definition + one example is enough for full marks."
    )

# ---------------- MAIN CHATBOT ----------------
# ...existing code...

# ---------------- MAIN CHATBOT ----------------
def get_response(user_input):
    user_input_clean = clean_text(user_input)

    # Dataset similarity
    user_vec = dataset_vectorizer.transform([user_input_clean])
    sim = cosine_similarity(user_vec, dataset_vectors)
    ds_index = sim.argmax()
    ds_score = sim[0][ds_index]

    # PDF similarity (safe)
    pdf_score = 0
    pdf_index = None
    if pdf_vectors is not None:
        pdf_vec = pdf_vectorizer.transform([user_input_clean])
        pdf_sim = cosine_similarity(pdf_vec, pdf_vectors)
        pdf_index = pdf_sim.argmax()
        pdf_score = pdf_sim[0][pdf_index]

    if max(ds_score, pdf_score) < 0.2:
        return "🤔 I’m not confident about this question. Please rephrase it."

    subject = detect_subject(user_input)

    # Cyber Security → PDF ONLY
    if subject == "Cyber Security" and pdf_index is not None:
        base_answer = pdf_chunks[pdf_index]
        source = "College Notes (PDF)"

    # Other subjects → Dataset
    elif ds_score >= 0.3:
        base_answer = answers[ds_index]
        source = "Prepared Dataset"

    else:
        return (
            "🤔 I could not find a confident answer for this question.\n\n"
            "Please ask from the syllabus topics or rephrase your question."
        )

    intent = detect_intent(user_input)

    if intent == "definition":
        response = base_answer
    else:
        response = expand_answer(base_answer)

    return (
        f"📚 **Subject:** {subject}\n"
        f"📖 **Source:** {source}\n\n"
        f"{response}\n\n"
        f"Ask if you want it simpler or more detailed."
    )