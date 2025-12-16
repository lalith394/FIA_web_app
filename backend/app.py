# backend/app.py
from flask import Flask, jsonify, request, send_from_directory
import shutil
from urllib.parse import urlparse
import json
from eval import infer_images, find_model_metadata
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS so frontend (Next.js) can talk to backend

@app.route('/')
def home():
    return jsonify(message="Flask backend is running!")


@app.route('/output/<path:filename>')
def serve_output(filename):
    # Serve generated output files (masks/features) from the backend output directory
    output_dir = os.path.join(os.getcwd(), 'output')
    return send_from_directory(output_dir, filename)


@app.route('/api/save_outputs', methods=['POST'])
def save_outputs():
    data = request.get_json() or {}
    urls = data.get('urls', [])
    dest_dir = data.get('dest_dir', '')

    if not urls:
        return jsonify({'ok': False, 'message': 'No files provided'}), 400

    output_root = os.path.join(os.getcwd(), 'output')
    saved_urls = []

    # normalize dest_dir to relative path inside output
    dest_dir = str(dest_dir or '').lstrip('/\\')
    dest_dir = dest_dir.strip()

    for url in urls:
        try:
            # if url is absolute URL, extract path
            if isinstance(url, str) and url.startswith('http'):
                parsed = urlparse(url)
                path = parsed.path
            else:
                path = url

            if path.startswith('/output/'):
                rel = path[len('/output/'):]
            else:
                rel = os.path.basename(path)

            src = os.path.join(output_root, rel)
            if not os.path.exists(src):
                # try fallback: maybe 'output/<rel>' already
                continue

            dest_base = os.path.join(output_root, dest_dir) if dest_dir else output_root
            os.makedirs(dest_base, exist_ok=True)

            filename = os.path.basename(src)
            dest = os.path.join(dest_base, filename)

            # avoid overwriting files
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dest):
                dest = os.path.join(dest_base, f"{base}_{counter}{ext}")
                counter += 1

            shutil.copy2(src, dest)
            rel_dest = os.path.relpath(dest, start=output_root).replace('\\', '/')
            saved_urls.append(f"{request.host_url.rstrip('/')}/output/{rel_dest}")
        except Exception as e:
            print(f"save_outputs: failed to copy {url}: {e}")
            continue

    return jsonify({'ok': True, 'saved': saved_urls})

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
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
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

            # detect model type via metadata if available (search all model dirs)
            try:
                models_root = os.path.join(os.getcwd(), 'models')
                mtype, _, _ = find_model_metadata(models_root, model)
            except FileNotFoundError:
                # fallback: no metadata found
                mtype = None

            # Run inference for segmentation and autoencoder model types (both produce image outputs)
            if mtype in ('segmentation', 'autoencoder'):
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
                # Prevent API callers from forcing the backend to save outputs
                # to arbitrary absolute filesystem locations. If the client
                # supplies an absolute path, map it into the backend's
                # output/ directory (use basename) so generated files are
                # always served under /output/ and accessible from the UI.
                if isinstance(out_dir_for_gen, str) and os.path.isabs(out_dir_for_gen):
                    # prefer a safe basename inside output/
                    out_dir_for_gen = os.path.basename(out_dir_for_gen) or model
                try:
                    print(f"Running inference for model={model}, files={saved_paths}, threshold={threshold}, batch_size={batch_size}, save_features={save_feats}")
                    generated = infer_images(model, saved_paths, threshold=threshold, out_dir=out_dir_for_gen, save_features=save_feats, batch_size=batch_size)
                    print(f"Inference returned generated: {generated}")
                    if not generated:
                        print("Warning: inference returned no generated files.")
                    else:
                        # convert absolute file paths into backend-accessible URLs under /output/<path>
                        generated_urls = []
                        output_root = os.path.join(os.getcwd(), 'output')
                        for p in generated:
                            try:
                                rel = os.path.relpath(p, start=output_root)
                            except Exception:
                                rel = os.path.basename(p)
                            rel = rel.replace('\\', '/')
                            # build absolute URL using request.host_url
                            base = request.host_url.rstrip('/')
                            generated_urls.append(f"{base}/output/{rel}")
                        generated = generated_urls
                except Exception as e:
                    return jsonify({'ok': False, 'message': f'Error running inference: {e}'}), 500

        return jsonify({'ok': True, 'message': 'Files received', 'saved': saved, 'model': model, 'output_dir': output_dir, 'config': config, 'generated': generated})
    except Exception as e:
        return jsonify({'ok': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)