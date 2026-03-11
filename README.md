# 🔍 Customer Insights Bot

**AI‑powered customer‑research agent** that surfaces pain points from raw feedback, notifies the right team on Slack, and generates interview scripts for AI‑led consumer interviews (ElevenLabs).

---

## Architecture

```
Customer Feedback ──► Analyzer (GPT‑4o) ──► Pain Points
                                              │
                                  ┌───────────┴───────────┐
                                  ▼                       ▼
                          Slack Notifier           Interview Generator
                        (webhook → #customer‑alert)   (GPT‑4o)
                                                        │
                                                        ▼
                                               Employee Dashboard
                                           (review / add questions / approve)
                                                        │
                                                        ▼
                                              AI Voice Interviews
                                               (ElevenLabs — next step)
```

## Quick Start

### 1. Install dependencies

```bash
cd customer-research-hackathon
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (required)
# SLACK_WEBHOOK_URL is pre‑filled
```

### 3. Run the API server

```bash
uvicorn main:app --reload
```

API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Run the employee dashboard

```bash
streamlit run dashboard.py
```

Dashboard at [http://localhost:8501](http://localhost:8501)

---

## How It Works

1. **Paste or upload** customer feedback in the dashboard (or `POST /analyze`).
2. The **AI Analyzer** (GPT‑4o) clusters feedback into pain points with severity, department, and keywords.
3. For each pain point, an **Interview Script** is auto‑generated with open‑ended questions designed for an AI voice agent.
4. A **Slack alert** is sent to `#customer‑alert` with a summary + link to the dashboard.
5. The employee can **review the script**, **add questions**, or **approve** it for launch.

## Tech Stack

| Layer | Tech |
|-------|------|
| AI/LLM | OpenAI GPT‑4o |
| Backend | FastAPI |
| Dashboard | Streamlit |
| Notifications | Slack Incoming Webhooks |
| Voice Interviews | ElevenLabs (planned) |