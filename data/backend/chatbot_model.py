import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

data = pd.read_csv("../data/study_data.csv")

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
    user_input = clean_text(user_input)
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, question_vectors)
    index = similarity.argmax()
    return answers[index]
