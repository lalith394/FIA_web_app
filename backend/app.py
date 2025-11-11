# backend/app.py
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS so frontend (Next.js) can talk to backend

@app.route('/')
def home():
    return jsonify(message="Flask backend is running!")

@app.route('/api/test', methods=['POST'])
def test_api():
    data = request.json
    name = data.get('name', 'Guest')
    return jsonify(message=f"Hello, {name}! Flask received your data successfully.")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)