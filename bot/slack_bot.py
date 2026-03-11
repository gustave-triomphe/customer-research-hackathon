"""Slack webhook helper – sends Block Kit messages to #customer-alert."""

import os, pathlib, requests
from dotenv import load_dotenv

# Load .env from the same directory as this file (works regardless of cwd)
_HERE = pathlib.Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")
WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")


def send_insight_alert(insight: str, questions: list[str]) -> bool:
    """Post the initial insight + proposed research questions to Slack."""
    q_text = "\n".join(f"• {q}" for q in questions)
    payload = {
        "text": f"🚨 {insight}",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🚨 Customer Insight Detected", "emoji": True},
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{insight}*",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "We will conduct more in-depth research into the root cause of this issue.\n\n"
                        "*Proposed interview questions:*\n"
                        f"{q_text}"
                    ),
                },
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "👉 Open the *Dashboard* to validate the research or add questions.",
                },
            },
        ],
    }
    try:
        r = requests.post(WEBHOOK, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"Slack error: {e}")
        return False


def send_validation_update(extra_questions: list[str] | None = None) -> bool:
    """Notify Slack that the ops manager validated (and optionally added questions)."""
    if extra_questions:
        q_text = "\n".join(f"• {q}" for q in extra_questions)
        body = (
            "✅ *Research validated by Ops Manager*\n\n"
            "*Additional questions added:*\n"
            f"{q_text}\n\n"
            "The research team will now proceed with the updated interview script."
        )
    else:
        body = (
            "✅ *Research validated by Ops Manager*\n\n"
            "No additional questions — the research team will proceed with the current script."
        )

    payload = {
        "text": "✅ Research validated",
        "blocks": [
            {"type": "divider"},
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": body},
            },
        ],
    }
    try:
        r = requests.post(WEBHOOK, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"Slack error: {e}")
        return False
