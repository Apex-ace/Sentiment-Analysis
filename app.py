import pickle
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, render_template_string

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
    print("❌ Pickle files not found! Please run your training script first.")
    exit()

# --- Ensure required NLTK data is downloaded ---
resources = ['stopwords', 'punkt', 'wordnet']
for resource in resources:
    try:
        nltk.data.find(f'corpora/{resource}' if resource != 'punkt' else f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)

# --- Preprocessing Functions ---
lemmatizer = WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words("english")) - {"not"}

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

def text_processing(text):
    """Processes cleaned text by removing stopwords and lemmatizing."""
    processed_text = []
    tokens = nltk.word_tokenize(text)
    for word in tokens:
        if word not in stopwords:
            processed_text.append(lemmatizer.lemmatize(word))
    return " ".join(processed_text)

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

        # Preprocess user input
        cleaned_review = text_cleaning(review_text)
        processed_review = text_processing(cleaned_review)

        # Vectorize text
        vectorized_review = tfidf_vectorizer.transform([processed_review])

        # Predict sentiment
        prediction_code = classifier.predict(vectorized_review)
        prediction = label_encoder.inverse_transform(prediction_code)[0]

    return render_template_string(HTML_TEMPLATE, prediction=prediction, review_text=review_text)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
