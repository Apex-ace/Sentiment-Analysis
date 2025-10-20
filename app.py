import pickle
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, render_template_string
import os

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Model and Preprocessing Artifacts ---
try:
    with open('classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
except FileNotFoundError:
    print(" Pickle files not found! Please run your training script first.")
    exit()

# --- Ensure required NLTK data is downloaded ---
nltk_resources = ['stopwords', 'punkt', 'wordnet', 'punkt_tab']
for res in nltk_resources:
    try:
        nltk.data.find(f'corpora/{res}' if res not in ['punkt', 'punkt_tab'] else f'tokenizers/{res}')
    except LookupError:
        nltk.download(res)

# --- Preprocessing Functions ---
lemmatizer = WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words("english")) - {"not"}

def text_cleaning(text):
    """Clean input text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    punc = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(punc)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\n', '', text)
    return text.strip()

def text_processing(text):
    """Remove stopwords and lemmatize."""
    tokens = nltk.word_tokenize(text)
    return " ".join(lemmatizer.lemmatize(word) for word in tokens if word not in stopwords)

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis</title>
</head>
<body>
    <h1>Musical Instrument Review Sentiment Analysis</h1>
    <form action="/" method="post">
        <label for="review">Enter a review:</label><br>
        <textarea name="review" id="review" rows="5" cols="50" required>{{ review_text }}</textarea><br><br>
        <input type="submit" value="Analyze Sentiment">
    </form>
    {% if prediction %}
        <p><strong>Predicted Sentiment: {{ prediction }}</strong></p>
    {% endif %}
</body>
</html>
"""

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    review_text = ""
    if request.method == 'POST':
        review_text = request.form['review']
        processed_review = text_processing(text_cleaning(review_text))
        vectorized_review = tfidf_vectorizer.transform([processed_review])
        prediction_code = classifier.predict(vectorized_review)
        prediction = label_encoder.inverse_transform(prediction_code)[0]
    return render_template_string(HTML_TEMPLATE, prediction=prediction, review_text=review_text)

# --- Run App ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render assigns a port automatically
    app.run(debug=True, host="0.0.0.0", port=port)
