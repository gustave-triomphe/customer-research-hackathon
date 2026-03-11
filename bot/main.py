"""FastAPI backend – Customer Insights Bot demo."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slack_bot import send_insight_alert, send_validation_update

app = FastAPI(title="Customer Insights Bot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Hardcoded demo data ───────────────────────────────────────────────────────

INSIGHT = "709 customers found that food was not fresh or tasty enough"

INITIAL_QUESTIONS = [
    "What dish did you order?",
    "What was wrong with it?",
]

# ── In-memory state ───────────────────────────────────────────────────────────

state = {
    "insight": INSIGHT,
    "questions": list(INITIAL_QUESTIONS),
    "validated": False,
    "slack_sent": False,
}


# ── Schemas ────────────────────────────────────────────────────────────────────

class AddQuestions(BaseModel):
    questions: list[str]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "Customer Insights Bot 🤖"}


@app.get("/state")
def get_state():
    """Return the current demo state."""
    return state


@app.post("/send-alert")
def send_alert():
    """Step 1 – Bot sends the insight + proposed questions to Slack."""
    ok = send_insight_alert(state["insight"], state["questions"])
    state["slack_sent"] = ok
    state["validated"] = False
    state["questions"] = list(INITIAL_QUESTIONS)  # reset
    return {"slack_sent": ok}


@app.post("/add-questions")
def add_questions(body: AddQuestions):
    """Ops manager adds extra questions to the script."""
    state["questions"].extend(body.questions)
    return {"questions": state["questions"]}


@app.post("/validate")
def validate():
    """Ops manager validates the research – notify Slack."""
    extra = [q for q in state["questions"] if q not in INITIAL_QUESTIONS]
    ok = send_validation_update(extra if extra else None)
    state["validated"] = True
    return {"validated": True, "slack_sent": ok}
