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
    print("❌ Pickle files not found! Please run your training script first.")
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
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Musical Instrument Review — Sentiment Analysis</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Playfair+Display:wght@600&display=swap" rel="stylesheet">
  <style>
    :root{
      --bg:#0f1724;
      --card:#0b1220;
      --glass: rgba(255,255,255,0.04);
      --accent1:#7c3aed; /* violet */
      --accent2:#06b6d4; /* teal */
      --muted: #94a3b8;
      --glass-border: rgba(255,255,255,0.06);
      color-scheme: dark;
    }
    *{box-sizing:border-box}
    html,body{height:100%;margin:0;font-family:Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; background:linear-gradient(180deg,#071029 0%, #081220 40%), radial-gradient(800px 400px at 10% 10%, rgba(124,58,237,0.08), transparent 20%), radial-gradient(600px 300px at 90% 90%, rgba(6,182,212,0.03), transparent 20%);}

    .wrap{min-height:100vh;display:flex;align-items:center;justify-content:center;padding:40px}
    .card{width:100%;max-width:880px;background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));border:1px solid var(--glass-border);border-radius:16px;padding:28px;box-shadow:0 10px 30px rgba(2,6,23,0.6);backdrop-filter: blur(6px);}

    .header{display:flex;gap:18px;align-items:center;margin-bottom:18px}
    .logo{width:64px;height:64px;border-radius:12px;display:grid;place-items:center;background:linear-gradient(135deg,var(--accent1),var(--accent2));box-shadow:0 6px 18px rgba(12,10,30,0.6);}
    .logo svg{width:40px;height:40px;filter:drop-shadow(0 3px 8px rgba(7,12,26,0.6));}
    h1{font-family:'Playfair Display', serif;color:white;margin:0;font-size:20px}
    p.lead{margin:0;color:var(--muted);font-size:13px}

    form{display:grid;grid-template-columns:1fr 340px;gap:18px;align-items:start}
    @media (max-width:880px){form{grid-template-columns:1fr} .aside{order:2}}

    .field{background:var(--card);padding:14px;border-radius:12px;border:1px solid var(--glass-border)}
    label{display:block;color:var(--muted);font-size:13px;margin-bottom:8px}
    textarea{width:100%;min-height:170px;resize:vertical;padding:12px;border-radius:8px;border:1px solid rgba(255,255,255,0.03);background:transparent;color:#e6eef8;font-size:14px;font-family:inherit}
    .actions{display:flex;gap:12px;align-items:center;margin-top:10px}
    .btn{display:inline-flex;align-items:center;gap:10px;padding:10px 14px;border-radius:10px;border:none;cursor:pointer;font-weight:600}
    .btn.primary{background:linear-gradient(90deg,var(--accent1),var(--accent2));color:white;box-shadow:0 8px 24px rgba(12,10,30,0.6)}
    .btn.ghost{background:transparent;border:1px solid var(--glass-border);color:var(--muted)}

    .aside{display:flex;flex-direction:column;gap:12px}
    .card-sm{background:var(--glass);padding:12px;border-radius:10px;border:1px solid var(--glass-border)}
    .result{padding:14px;border-radius:10px;background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));border:1px solid rgba(255,255,255,0.03);min-height:78px;display:flex;align-items:center;justify-content:center;font-weight:700}
    .result.positive{color:#10b981}
    .result.negative{color:#ef4444}
    .result.neutral{color:#f59e0b}

    footer{margin-top:16px;color:var(--muted);font-size:13px;display:flex;justify-content:space-between;align-items:center}

    /* small helper */
    .mini{font-size:12px;color:var(--muted)}

    /* subtle focus styles */
    textarea:focus{outline:none;box-shadow:0 0 0 4px rgba(124,58,237,0.06);border-color:rgba(124,58,237,0.6)}

    /* spinner */
    .spinner{width:18px;height:18px;border-radius:50%;border:3px solid rgba(255,255,255,0.08);border-top-color:rgba(255,255,255,0.6);animation:spin 1s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="header">
        <div class="logo" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 12c0-3.866 3.582-7 8-7s8 3.134 8 7-3.582 7-8 7S3 15.866 3 12z" fill="white" opacity="0.08"/>
            <path d="M12 7c2.761 0 5 2.015 5 5s-2.239 5-5 5-5-2.015-5-5 2.239-5 5-5z" fill="white"/>
          </svg>
        </div>
        <div>
          <h1>Musical Instrument Review — Sentiment Analysis</h1>
          <p class="lead">Paste any review and get an instant sentiment prediction (server-powered).</p>
        </div>
      </div>

      <form action="/" method="post" onsubmit="onSubmit(event)">
        <div class="field">
          <label for="review">Enter a review</label>
          <textarea name="review" id="review" rows="6" required placeholder="e.g. The acoustic guitar has a warm tone but the build quality is average...">{{ review_text }}</textarea>
          <div class="actions">
            <button type="submit" class="btn primary" id="analyzeBtn">
              <span id="btnText">Analyze Sentiment</span>
              <span id="btnSpinner" style="display:none;margin-left:6px" aria-hidden="true" class="spinner"></span>
            </button>
            <button type="button" class="btn ghost" onclick="clearForm()">Clear</button>
            <div style="margin-left:auto" class="mini">Model: <strong>server-side</strong></div>
          </div>
        </div>

        <aside class="aside">
          <div class="card-sm">
            <label class="mini">Quick tips</label>
            <ul class="mini" style="margin:8px 0 0;padding-left:18px;line-height:1.5;color:var(--muted)">
              <li>Be specific — include instrument, tone, playability.</li>
              <li>Short and direct sentences give clearer predictions.</li>
            </ul>
          </div>

          <div class="card-sm">
            <label class="mini">Predicted Sentiment</label>
            <div class="result {{ 'positive' if prediction=='Positive' else 'negative' if prediction=='Negative' else 'neutral' if prediction=='Neutral' else '' }}">
              {% if prediction %}
                {{ prediction }}
              {% else %}
                <span class="mini">No prediction yet — submit a review to analyze.</span>
              {% endif %}
            </div>
          </div>

          
          
        </aside>
      </form>

      
    </div>
  </div>

  <script>
    function onSubmit(e){
      // show spinner and let the form submit normally (server will respond with template)
      const btn = document.getElementById('analyzeBtn');
      const text = document.getElementById('btnText');
      const spinner = document.getElementById('btnSpinner');
      text.textContent = 'Analyzing...';
      spinner.style.display = 'inline-block';
      btn.disabled = true;
    }
    function clearForm(){
      document.getElementById('review').value = '';
    }
  </script>
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
    app.run(debug=True)
