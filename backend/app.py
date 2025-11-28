# backend/app.py
from flask import Flask, jsonify, request
import json
from eval import infer_images
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
    data = request.json or {}
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
    saved_paths = []
    try:
        for f in files:
            # preserve nested relative path segments sent by the client
            orig_name = f.filename or "unnamed"
            # normalize separators and sanitize each segment
            parts = orig_name.replace('\\', '/').split('/')
            safe_parts = [secure_filename(p) for p in parts if p and p != '.' ]
            dest = os.path.join(save_dir, *safe_parts)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            f.save(dest)
            saved.append("/".join(safe_parts))
            saved_paths.append(dest)

        # If a model was supplied, attempt inference for segmentation models
        generated = None
        if model:
            # parse config JSON if provided
            try:
                cfg = json.loads(config) if config else {}
            except Exception:
                cfg = {}

            # detect model type via metadata if available
            try:
                # Prefer segmentation model metadata location
                meta_path_seg = os.path.join(os.getcwd(), 'models', 'segmentation', model, 'metadata.json')
                meta_path_root = os.path.join(os.getcwd(), 'models', model, 'metadata.json')

                if os.path.exists(meta_path_seg):
                    with open(meta_path_seg, 'r', encoding='utf8') as mf:
                        meta = json.load(mf)
                        mtype = meta.get('type')
                elif os.path.exists(meta_path_root):
                    with open(meta_path_root, 'r', encoding='utf8') as mf:
                        meta = json.load(mf)
                        mtype = meta.get('type')
                else:
                    # If no metadata exists, guess 'segmentation' if model folder exists under models/segmentation
                    seg_folder = os.path.join(os.getcwd(), 'models', 'segmentation', model)
                    if os.path.isdir(seg_folder):
                        mtype = 'segmentation'
                    else:
                        mtype = None
            except Exception:
                mtype = None

            if mtype == 'segmentation':
                # Helper to parse booleans from form/config values
                def parse_bool(v):
                    if isinstance(v, bool):
                        return v
                    if v is None:
                        return False
                    return str(v).lower() in ("1", "true", "yes", "on")

                # threshold can be passed inside config (as JSON) or as form field
                threshold = float(cfg.get('threshold', request.form.get('threshold', 0.5)))

                # batch size may be provided as batch_size or batchSize
                batch_size = int(cfg.get('batch_size', cfg.get('batchSize', request.form.get('batch_size', 1))))

                save_feats = parse_bool(cfg.get('save_features', cfg.get('saveFeatures', request.form.get('save_features', False))))

                out_dir_for_gen = output_dir or model
                try:
                    print(f"Running inference for model={model}, files={saved_paths}, threshold={threshold}, batch_size={batch_size}, save_features={save_feats}")
                    generated = infer_images(model, saved_paths, threshold=threshold, out_dir=out_dir_for_gen, save_features=save_feats, batch_size=batch_size)
                    print(f"Inference returned generated: {generated}")
                    if not generated:
                        print("Warning: inference returned no generated files.")
                except Exception as e:
                    return jsonify({'ok': False, 'message': f'Error running inference: {e}'}), 500

        return jsonify({'ok': True, 'message': 'Files received', 'saved': saved, 'model': model, 'output_dir': output_dir, 'config': config, 'generated': generated})
    except Exception as e:
        return jsonify({'ok': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)