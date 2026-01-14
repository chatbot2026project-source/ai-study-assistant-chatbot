import pandas as pd
import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "study_data.csv")

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

# ---------------- TF-IDF ----------------
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(cleaned_questions)

# ---------------- INTENT ----------------
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

# ---------------- SUBJECT ----------------
def detect_subject(q):
    q = q.lower()
    if any(w in q for w in ["deadlock", "paging", "cpu", "process"]):
        return "Operating Systems"
    if any(w in q for w in ["dbms", "sql", "normalization"]):
        return "DBMS"
    if any(w in q for w in ["python", "class", "inheritance"]):
        return "Programming"
    if any(w in q for w in ["cyber", "security", "attack", "malware"]):
        return "Cyber Security"
    return "General Studies"

# ---------------- EXPAND ANSWER ----------------
def expand_answer(answer):
    return (
        f"{answer}\n\n"
        f"📌 **In simple terms:**\n"
        f"This is an important concept you should understand clearly.\n\n"
        f"🎯 **Exam Tip:**\n"
        f"Write definition + one example for full marks."
    )

# ---------------- MAIN FUNCTION ----------------
def get_response(user_input):
    user_input_clean = clean_text(user_input)
    user_vector = vectorizer.transform([user_input_clean])

    similarity = cosine_similarity(user_vector, question_vectors)
    best_index = similarity.argmax()
    confidence = similarity[0][best_index]

    if confidence < 0.3:
        return (
            "🤔 I’m not confident about this question.\n\n"
            "Please ask from your syllabus topics or rephrase."
        )

    base_answer = answers[best_index]
    intent = detect_intent(user_input)
    subject = detect_subject(user_input)

    if intent == "definition":
        response = base_answer
    else:
        response = expand_answer(base_answer)

    return (
        f"📚 **Subject:** {subject}\n"
        f"📖 **Source:** Prepared Dataset\n\n"
        f"{response}"
    )
