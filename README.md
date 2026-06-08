# RAGbot — PDF-Grounded AI Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot. Upload any PDF; the backend indexes it with ChromaDB + Sentence Transformers; Groq LLM answers questions strictly from the document. Leads are captured to Google Sheets and notified via email.

---

## Architecture

```
frontend/index.html          ← Single-file luxury frontend (WebGL, custom cursor)
backend/
  main.py                    ← FastAPI app — all routes, CORS, scheduler
  chat.py                    ← Groq call logic + lead detection
  rag.py                     ← PDF extract → chunk → embed → ChromaDB
  models.py                  ← Pydantic request/response models
  config.py                  ← Settings (loaded from .env)
  requirements.txt           ← All Python deps
  .env.example               ← Copy → .env and fill in
render.yaml                  ← One-click Render deploy config
```

---

## Local Development

### 1. Prerequisites
- Python 3.10+
- A Groq API key (free at https://console.groq.com)

### 2. Backend setup

```bash
cd backend
cp .env.example .env
# Edit .env — at minimum set GROQ_API_KEY, EMAIL_* vars
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Frontend
Open `frontend/index.html` directly in a browser, or serve it:

```bash
cd frontend
python -m http.server 3000
# Then open http://localhost:3000
```

Make sure `API_BASE` in `index.html` matches your backend URL (default: `http://localhost:8000`).

---

## Google Sheets Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com) → APIs & Services → Enable **Google Sheets API** and **Google Drive API**
2. Create a **Service Account** → download the JSON key → save as `backend/google-credentials.json`
3. Create a Google Sheet named exactly **"RAG Chatbot Leads"**
4. Share that sheet with the service account email (Editor permission)
5. Add headers to row 1: `Name | Phone | Session ID | Query`

---

## Render Deployment

### One-click (render.yaml)
1. Push the repo to GitHub
2. In Render: **New → Blueprint** → connect your repo → Render reads `render.yaml`
3. In the Render dashboard → **Environment** → add all secret vars from `.env.example`
4. Upload `google-credentials.json` via Render's Secret Files feature (or embed as env var)
5. After deploy, update `API_BASE` in `frontend/index.html` to your Render URL

### Manual
- Runtime: Python 3.11
- Root directory: `backend`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Add all env vars manually in the dashboard

> **Note on persistence**: Render's free tier has an ephemeral filesystem — ChromaDB collections are lost on redeploy. Upgrade to a paid plan and uncomment the `disk:` section in `render.yaml` for persistent vector storage.

---

## API Reference

| Method | Endpoint                     | Description                          |
|--------|------------------------------|--------------------------------------|
| GET    | `/health`                    | Server health check                  |
| POST   | `/upload`                    | Upload + process a PDF               |
| POST   | `/chat`                      | Send a message, get RAG response     |
| POST   | `/lead`                      | Submit a lead (name + phone)         |
| GET    | `/session/{id}`              | Validate a session ID                |
| DELETE | `/session/{id}/history`      | Clear conversation history           |

---

## Environment Variables

See `.env.example` for the full list. Minimum required:

```
GROQ_API_KEY=...
EMAIL_SENDER=...
EMAIL_PASSWORD=...
EMAIL_RECIPIENT=...
```

---

## Rate Limits
- `/upload` — 10 requests/minute per IP
- `/chat`   — 30 requests/minute per IP
- `/lead`   — 5 requests/minute per IP
