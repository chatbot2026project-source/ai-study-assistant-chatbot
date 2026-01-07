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

def get_response(user_input):
    user_input_clean = clean_text(user_input)
    user_vector = vectorizer.transform([user_input_clean])

    similarity = cosine_similarity(user_vector, question_vectors)
    best_match_index = similarity.argmax()
    confidence = similarity[0][best_match_index]

    answer = answers[best_match_index]

    # ChatGPT-style response template
    if confidence < 0.2:
        return (
            "I’m not fully sure about this question yet. "
            "Please try asking in a different way or check your subject notes."
        )

    return (
        f"Here is a simple explanation:\n\n"
        f"{answer}\n\n"
        f"Let me know if you want this explained in more detail."
    )

