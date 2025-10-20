import os
import json
import requests
import time
from flask import Flask, request, render_template_string

# --- Flask App Initialization ---
app = Flask(__name__)
API_KEY = "AIzaSyDGisna7FUkrp-b2y8I7C6W1z7EDSaRgyI"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

# Define the expected JSON structure for reliable output
SENTIMENT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "sentiment": {
            "type": "STRING",
            "description": "The determined sentiment: 'Positive', 'Negative', or 'Neutral'."
        },
        "confidence_percentage": {
            "type": "INTEGER",
            "description": "The model's confidence in the sentiment prediction, represented as an integer percentage from 0 to 100."
        }
    },
    "propertyOrdering": ["sentiment", "confidence_percentage"]
}

def get_sentiment_from_gemini(review_text):
    """
    Calls the Gemini API to perform structured sentiment analysis on the text.
    Implements exponential backoff for resilience.
    """
    if not review_text.strip():
        return None, "Review text is empty."

    # UPDATED PROMPT: Removed mention of generating a summary.
    system_prompt = "You are an expert sentiment analysis bot for musical instrument reviews. Your task is to analyze the provided review text, determine the overall sentiment ('Positive', 'Negative', or 'Neutral'), and provide a confidence percentage (0-100) for your prediction. Output your response STRICTLY as a JSON object."

    user_query = f"Analyze the sentiment of this musical instrument review: \"{review_text}\""

    payload = {
        "contents": [{
            "parts": [{"text": user_query}]
        }],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": SENTIMENT_SCHEMA
        }
    }

    # Exponential backoff parameters
    max_retries = 3
    base_delay = 1

    for attempt in range(max_retries):
        try:
            headers = {'Content-Type': 'application/json'}
            # Note: Canvas handles API_KEY injection automatically if it's empty.
            response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, json=payload, timeout=30)
            response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            
            # Parse the structured response from the model
            json_text = result['candidates'][0]['content']['parts'][0]['text']
            sentiment_data = json.loads(json_text)
            
            return sentiment_data, None

        except requests.exceptions.RequestException as e:
            # Handle connection errors, timeouts, and bad HTTP status codes
            if attempt < max_retries - 1 and (response.status_code == 429 or response.status_code >= 500):
                delay = base_delay * (2 ** attempt) + (0.5 * random.random())
                print(f"API Error (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
            else:
                print(f"Fatal API Error after {attempt+1} attempts: {e}")
                return None, f"Could not connect to AI API: {e}" # Updated error message
        except (json.JSONDecodeError, KeyError) as e:
            # Handle malformed JSON response from the model
            print(f"Error parsing AI response: {e}. Raw response: {response.text}") # Updated error message
            return None, "Error parsing model response. Please try again."

    return None, "AI API failed after multiple retries." # Updated error message

# --- HTML Template (Using Tailwind CSS via CDN for aesthetics) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analyzer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-gray-50 min-h-screen flex items-center justify-center p-4">
    <div class="bg-white p-8 rounded-xl shadow-2xl w-full max-w-lg">
        <h1 class="text-3xl font-extrabold text-indigo-700 mb-6 border-b-2 border-indigo-100 pb-2">
            AI Review Sentiment Analysis
        </h1>
        <p class="text-gray-600 mb-6">
            Powered by AI, this tool analyzes your review and returns a structured sentiment result.
        </p>

        <form action="/" method="post">
            <label for="review" class="block text-sm font-medium text-gray-700 mb-2">
                Enter Musical Instrument Review:
            </label>
            <textarea name="review" id="review" rows="6"
                      class="w-full p-3 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500 transition duration-150 ease-in-out"
                      placeholder="e.g., The guitar has an unbelievably warm tone, but the tuning pegs feel cheap.">{{ review_text }}</textarea>
            <button type="submit"
                    class="mt-4 w-full px-4 py-2 bg-indigo-600 text-white font-semibold rounded-lg shadow-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition duration-150 ease-in-out">
                Analyze Sentiment
            </button>
        </form>

        {% if result %}
            {% set sentiment_class = {
                'Positive': 'bg-green-100 text-green-800 border-green-400',
                'Negative': 'bg-red-100 text-red-800 border-red-400',
                'Neutral': 'bg-yellow-100 text-yellow-800 border-yellow-400',
            }.get(result.sentiment, 'bg-gray-100 text-gray-800 border-gray-400') %}

            <div class="mt-8 p-4 rounded-xl border-l-4 {{ sentiment_class }} shadow-inner">
                <p class="text-lg font-bold mb-4 flex items-center">
                    <span class="mr-2">{% if result.sentiment == 'Positive' %}😊{% elif result.sentiment == 'Negative' %}😞{% else %}😐{% endif %}</span>
                    Predicted Sentiment: <span class="ml-2 px-3 py-1 rounded-full text-sm font-medium {{ sentiment_class }}">{{ result.sentiment }}</span>
                </p>
                
                {# Confidence Display Block #}
                <div class="pt-3 border-t border-gray-200">
                    <p class="text-sm font-medium text-gray-700 mb-1">Confidence Score: {{ result.confidence_percentage }}%</p>
                    <div class="w-full bg-gray-200 rounded-full h-2.5">
                        <div class="h-2.5 rounded-full transition-all duration-500 ease-in-out"
                             style="width: {{ result.confidence_percentage }}%; 
                                    background-color: {% if result.confidence_percentage >= 80 %}#10B981{% elif result.confidence_percentage >= 50 %}#F59E0B{% else %}#EF4444{% endif %};">
                        </div>
                    </div>
                </div>
                {# End Confidence Display Block #}

            </div>
        {% elif error %}
            <div class="mt-8 p-4 rounded-xl border-l-4 border-red-500 bg-red-100 text-red-800 shadow-inner">
                <p class="text-lg font-bold mb-2">Error</p>
                <p class="text-sm">{{ error }}</p>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment_data = None
    error_message = None
    review_text = request.form.get('review', '') if request.method == 'POST' else ''

    if request.method == 'POST':
        sentiment_data, error_message = get_sentiment_from_gemini(review_text)

    return render_template_string(
        HTML_TEMPLATE,
        result=sentiment_data,
        error=error_message,
        review_text=review_text
    )

# --- Run App ---
if __name__ == '__main__':
    # Add a simple random module for backoff jitter, which is a good practice.
    import random 
    port = int(os.environ.get("PORT", 9898))
    app.run(debug=True, host="0.0.0.0", port=port)
