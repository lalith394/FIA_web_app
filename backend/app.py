# backend/app.py
from flask import Flask, jsonify, request
import os
from werkzeug.utils import secure_filename
from datetime import datetime
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


@app.route('/api/generate', methods=['POST'])
def generate_api():
    # Accept multipart/form-data with files under 'images' and fields model, output_dir, config
    files = request.files.getlist('images')

    model = request.form.get('model')
    output_dir = request.form.get('output_dir')
    config = request.form.get('config')

    if not files:
        return jsonify({'ok': False, 'message': 'No files uploaded'}), 400

    # Create an uploads directory with timestamp
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(os.getcwd(), 'uploads', ts)
    os.makedirs(save_dir, exist_ok=True)

    saved = []
    try:
        for f in files:
            # preserve filename/path if provided in client via the filename parameter
            filename = secure_filename(f.filename)
            dest = os.path.join(save_dir, filename)
            f.save(dest)
            saved.append(filename)

        return jsonify({'ok': True, 'message': 'Files received', 'saved': saved, 'model': model, 'output_dir': output_dir, 'config': config})
    except Exception as e:
        return jsonify({'ok': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)