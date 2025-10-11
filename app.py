from flask import Flask, request, jsonify, render_template
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Initialize Flask App and Load Models ---
app = Flask(__name__)

# Load the saved models
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("Models loaded successfully!")
# Catch multiple potential errors during file loading
except (FileNotFoundError, pickle.UnpicklingError) as e:
    print(f"CRITICAL ERROR: Could not load model files. {e}")
    print("Please ensure 'model.pkl' and 'vectorizer.pkl' are in the correct directory and are not corrupted.")
    model = None
    vectorizer = None

# --- Text Preprocessing Function (MUST be identical to training) ---
# Download NLTK data if not present
try:
    stopwords.words('english')
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    print("Downloading NLTK data (stopwords, wordnet)...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("NLTK data downloaded.")

# Create the stop words list (with sentiment words kept)
stop_words = set(stopwords.words('english'))
words_to_keep = {'good', 'bad', 'not', 'no', 'great', 'poor', 'best', 'worst'}
stop_words = stop_words - words_to_keep
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Using raw strings (r'...') to avoid syntax warnings
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    # Remove emojis and other non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Explicitly check if the model objects are None
    if model is None or vectorizer is None:
        return jsonify({'error': 'Models not loaded. Check server logs for errors.'}), 500

    text = request.form['review']
    processed_text = clean_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    
    # --- KEY CHANGE: Get prediction probabilities ---
    probabilities = model.predict_proba(vectorized_text)[0]
    class_labels = model.classes_
    
    # Create a dictionary of class labels to their probabilities (formatted as percentages)
    prob_dict = {label: f"{prob:.2%}" for label, prob in zip(class_labels, probabilities)}
    
    # Get the final prediction by finding the class with the highest probability
    prediction = class_labels[probabilities.argmax()]

    # --- DEBUGGING LOG: This will print to your terminal ---
    print("\n--- New Prediction ---")
    print(f"RAW TEXT: '{text}'")
    print(f"PROCESSED TEXT: '{processed_text}'")
    print(f"PROBABILITIES: {prob_dict}") # This is the crucial line for debugging
    print(f"PREDICTION: {prediction.upper()}")
    print("----------------------")

    return render_template(
        'index.html', 
        prediction_text=f'Sentiment: {prediction.capitalize()}', 
        review_text=text,
        probabilities=prob_dict # Pass probabilities to the template
    )

if __name__ == '__main__':
    app.run(debug=True)