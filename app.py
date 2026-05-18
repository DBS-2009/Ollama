from flask import Flask, render_template, request, jsonify
import os
import requests

app = Flask(__name__)

OLLAMA_API_URL = os.environ.get('OLLAMA_API_URL', '').strip() or os.environ.get('OLLAMAAPIURL', '').strip()
OLLAMA_API_KEY = os.environ.get('OLLAMA_API_KEY', '').strip() or os.environ.get('OLLAMAKEY', '').strip()
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'gemma4:e4b')

if not OLLAMA_API_URL:
    OLLAMA_API_URL = 'http://localhost:11434/api/generate'
    app.logger.warning('OLLAMA_API_URL is not set. Using localhost fallback for local development.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not OLLAMA_API_URL:
        return jsonify({"response": "AI service endpoint is not configured. Please set OLLAMA_API_URL (or OLLAMAAPIURL)."})

    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

    response = None
    for attempt in range(2):
        try:
            response = requests.post(OLLAMA_API_URL,
                                     json={"model": OLLAMA_MODEL, "prompt": user_input, "stream": False},
                                     headers=headers,
                                     timeout=(5, 120))
            break
        except requests.exceptions.Timeout:
            if attempt == 1:
                return jsonify({"response": "Sorry, the AI service timed out. Please try again."})
        except requests.exceptions.ConnectionError:
            return jsonify({"response": "Sorry, I could not reach the AI service. Please try again."})

    if response is None:
        return jsonify({"response": "Sorry, I couldn't connect to the AI service."})

    response_json = response.json()
    ai_response = None

    if isinstance(response_json, dict):
        ai_response = response_json.get('response')
        if not ai_response:
            ai_response = response_json.get('generated_text')
        if not ai_response and response_json.get('output'):
            output = response_json.get('output')
            if isinstance(output, list) and len(output) > 0:
                first = output[0]
                if isinstance(first, dict):
                    ai_response = first.get('content') or first.get('text')
                    if isinstance(ai_response, list) and len(ai_response) > 0:
                        ai_response = ai_response[0].get('text') if isinstance(ai_response[0], dict) else ai_response[0]
            elif isinstance(output, str):
                ai_response = output

    if not ai_response:
        ai_response = "Sorry, I couldn't generate a response."

    return jsonify({"response": ai_response})

if __name__ == '__main__':
    app.run(debug=True)
