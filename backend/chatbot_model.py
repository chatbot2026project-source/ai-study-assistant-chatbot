import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "study_data.csv")

data = pd.read_csv(DATA_PATH)

questions = data["question"].values
answers = data["answer"].values

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

cleaned_questions = [clean_text(q) for q in questions]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(cleaned_questions)

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

def detect_subject(question):
    q = question.lower()

    if "deadlock" in q or "paging" in q or "cpu" in q:
        return "Operating Systems"
    elif "sql" in q or "dbms" in q or "normalization" in q:
        return "DBMS"
    elif "python" in q or "class" in q or "inheritance" in q:
        return "Programming"
    else:
        return "General Studies"
    
def get_response(user_input):
    user_input_clean = clean_text(user_input)
    user_vector = vectorizer.transform([user_input_clean])

    similarity = cosine_similarity(user_vector, question_vectors)
    best_match_index = similarity.argmax()
    confidence = similarity[0][best_match_index]

    if confidence < 0.2:
        return (
            "I'm not confident about this question yet 🤔.\n"
            "Please try rephrasing it or ask from your syllabus topics."
        )

    answer = answers[best_match_index]
    intent = detect_intent(user_input)
    subject = detect_subject(user_input)

    if intent == "definition":
        return f"📘 **Definition:**\n{answer}"

    elif intent == "explanation":
        return (
            f"🧠 **Detailed Explanation:**\n"
            f"{answer}\n\n"
            f"👉 This concept is very important for exams."
        )

    elif intent == "reason":
        return (
            f"❓ **Why does this happen?**\n"
            f"{answer}\n\n"
            f"This occurs due to system design and resource usage."
        )

    elif intent == "comparison":
        return (
            f"🔍 **Comparison Insight:**\n"
            f"{answer}\n\n"
            f"If you want, I can explain with a table."
        )

    else:
        return (
            f"🤖 **Here’s what you need to know:**\n"
            f"{answer}\n\n"
            f"Ask me if you want a simpler explanation."
        )




