# FIA Web App 🩺

A lightweight image processing and model inference web app with a React/Next.js frontend and a Python (Flask + PyTorch) backend. Designed for tasks such as retinal vessel segmentation, autoencoder reconstructions, and classification experiments.

---

## 🚀 Highlights

- Backend: Python + Flask using PyTorch models (UNet for segmentation, AutoEncoders, and custom variants).
- Frontend: Next.js (App Router) UI for uploading images, configuring models, and viewing/saving outputs.
- Metadata-driven model discovery: models are discovered using `metadata.json` placed under `models/<type>/<model>/metadata.json`.
- Safety: client-supplied absolute output directories are normalized to safe paths inside `output/` to avoid arbitrary filesystem writes.
- Outputs: segmentation produces two artifacts per image: a strict binary mask (`*_mask.png`) and a raw probability map (`*_mask_raw.png`); UI displays the binary mask by default.

---

## 📁 Repository Layout

Top-level important folders:

- `backend/` — Flask API, model loading and inference helpers (`eval.py`).
- `frontend/` — Next.js app and UI components.
- `models/` — Model checkpoints and `metadata.json` descriptors.
- `uploads/` — Temporary uploaded inputs from the frontend API.
- `output/` — Generated outputs that are served under `/output/<path>`.

---

## 🔧 Quickstart (Development)

### Backend (Python / PyTorch)

1. Create a Python virtual environment and install dependencies (from `backend/`):

```bash
cd backend
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# or on macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

2. Start the Flask backend (development):

```bash
cd backend
python app.py
#or
flask --app app.py run
```

The backend exposes endpoints like `/api/generate` and serves artifacts under `/output/<path>`.

### Frontend (Next.js)

1. Install frontend dependencies and start dev server:

```bash
cd frontend
npm install
npm run dev
# or
# pnpm install && pnpm dev
```

2. Visit `http://localhost:3000` to open the UI.

---

## 🧪 Running Tests

Backend tests are implemented with `pytest`:

```bash
cd backend
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```

Tests cover the `/api/generate` flows for segmentation and autoencoder models and validate that outputs are saved and served correctly.

---

## 🧠 API Overview

### POST /api/generate

Accepts multipart/form-data with fields:
- `images` — one or more uploaded image files
- `model` — model name (must match a folder under `models/` with a `metadata.json`)
- `output_dir` — (optional) where to store outputs (string). Absolute paths provided by clients are mapped into `output/<basename>` for safety.
- `config` — optional JSON string with inference options e.g. `{ "threshold": 0.5, "batchSize": 1, "saveFeatures": true }`

Returns JSON with `generated` — a list of publicly-accessible URLs under `/output/`.

Notes for segmentation:
- The API returns two files per input when running segmentation: `*_mask.png` (binary 0/255 B/W) and `*_mask_raw.png` (grayscale probability map).
- The UI shows the binary mask by default to ensure crisp black-and-white visualizations.

### POST /api/save_outputs

Copies a list of `/output/...` URLs into a destination folder (within `output/`) and returns new `/output/...` URLs for the saved copies.

---

## 🖼️ Output conventions

- Segmentation outputs:
  - `<stem>_mask.png` — strict black & white mask (values 0 or 255).
  - `<stem>_mask_raw.png` — raw model prediction (probabilities / scores), saved as a grayscale image.
- Autoencoder reconstructions:
  - `<stem>.png` — single-channel grayscale reconstruction image.
- Feature maps (if `save_features=true`) are stored under `<stem>_d4_layer/channel_*.png`.

These files are saved under `output/<output_dir>` and served at `/output/<path>`.

---

## ⚠️ Security & Notes

- The backend prevents writing to arbitrary absolute paths provided by clients; absolute `output_dir` values are normalized to their basename inside `output/` (e.g., `C:\data\x` → `output/x`). If you require trusted absolute paths, implement a server-side allowlist.
- Models must include a `metadata.json` listing the model `type` and, when applicable, `parameters.num_channels` and `parameters.resolution` to ensure proper preprocessing and output sizing.

---

## 🛠 Development Tips

- Add models under `models/<type>/<name>/` with `metadata.json` and one or more `.pth` weight files. The loader prefers `<name>.pth` and otherwise picks the first `.pth` found.
- Use `backend/demo_infer.py` (if present) to exercise `infer_images()` locally using image files or folders.

---

## ✅ Contributing

Contributions are welcome — open issues and PRs for bug fixes and improvements. Please:

1. Fork the repo
2. Create a feature branch
3. Add tests for new behavior
4. Open a PR describing the change

---

## 📜 License & Credits

This project contains models and code built for research and demo purposes — please consult included `metadata.json` and model licenses for usage constraints. Add an appropriate `LICENSE` file to the repo if needed.

---

If you want, I can also add:
- a short example showing how to call `/api/generate` with `curl`,
- a small demo script (CLI) that uses a PyTorch DataLoader to run batch inference and save outputs,
- or a `Makefile` / dev scripts to simplify running backend + frontend in development.

If you'd like any of these additions, tell me which and I'll add them. 💡

