import pandas as pd
import numpy as np
import re
import string
import nltk
import pickle
import warnings
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# --- Ensure required NLTK data is downloaded ---
resources = ['stopwords', 'punkt', 'wordnet']
for resource in resources:
    try:
        nltk.data.find(f'corpora/{resource}' if resource != 'punkt' else f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)

# --- Preprocessing Functions ---

def labelling(rows):
    """Assigns a sentiment label based on the 'overall' rating."""
    if rows["overall"] > 4.0:
        return "Positive"
    elif rows["overall"] < 2.0:
        return "Negative"
    else:
        return "Neutral"

def text_cleaning(text):
    """Cleans the input text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    punc = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(punc)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\n', '', text)
    return text.strip()

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words("english")) - {"not"}

def text_processing(text):
    """Processes cleaned text by removing stopwords and lemmatizing."""
    processed_text = []
    tokens = nltk.word_tokenize(text)
    for word in tokens:
        if word not in stopwords:
            processed_text.append(lemmatizer.lemmatize(word))
    return " ".join(processed_text)

# --- Main script execution ---

def create_and_save_model():
    """Loads data, preprocesses it, trains a model, and saves the artifacts."""
    print("Loading dataset...")
    dataset = pd.read_csv("Instruments_Reviews.csv")

    print("Preprocessing data...")
    dataset['reviewText'] = dataset['reviewText'].fillna("")
    dataset['summary'] = dataset['summary'].fillna("")
    dataset["reviews"] = dataset["reviewText"] + " " + dataset["summary"]

    dataset["sentiment"] = dataset.apply(labelling, axis=1)

    print("Cleaning and processing text...")
    dataset["reviews"] = dataset["reviews"].apply(text_cleaning)
    dataset["reviews"] = dataset["reviews"].apply(text_processing)

    # Encode target variable
    encoder = LabelEncoder()
    dataset["sentiment"] = encoder.fit_transform(dataset["sentiment"])

    print("Vectorizing text with TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf_vectorizer.fit_transform(dataset["reviews"])
    y = dataset["sentiment"]

    print("Balancing dataset with SMOTE...")
    balancer = SMOTE(random_state=42)
    X_final, y_final = balancer.fit_resample(X, y)

    print("Training Logistic Regression model...")
    classifier = LogisticRegression(random_state=42, C=6866.488450042998, penalty='l2', max_iter=1000)
    classifier.fit(X_final, y_final)

    print("Saving model and vectorizer to pickle files...")
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open('classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    print("\n✅ Model creation complete!")
    print("Files created: tfidf_vectorizer.pkl, classifier.pkl, label_encoder.pkl")

    # Optional sample prediction check
    sample = ["This guitar is absolutely amazing, great sound quality!"]
    sample_vec = tfidf_vectorizer.transform(sample)
    pred = encoder.inverse_transform(classifier.predict(sample_vec))
    print("\nSample Prediction:", pred[0])


if __name__ == '__main__':
    create_and_save_model()
