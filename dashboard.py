"""Streamlit dashboard – Ops Manager view."""

import streamlit as st
import requests

API = "http://localhost:8000"

st.set_page_config(page_title="Customer Insights Bot", page_icon="🔍", layout="centered")


def api(method, path, **kw):
    try:
        r = getattr(requests, method)(f"{API}{path}", timeout=15, **kw)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ API not reachable – run `uvicorn main:app --reload`")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Header ─────────────────────────────────────────────────────────────────────

st.title("🔍 Customer Insights Bot")
st.caption("Demo – Ops Manager Dashboard")

st.divider()

# ── Step 1: Send the insight alert ─────────────────────────────────────────────

st.subheader("Step 1 · Send insight to Slack")
st.info("**Insight:** 709 customers found that food was not fresh or tasty enough")

if st.button("🚀 Send alert to #customer-alert", type="primary", use_container_width=True):
    with st.spinner("Sending to Slack…"):
        res = api("post", "/send-alert")
    if res and res.get("slack_sent"):
        st.success("✅ Alert sent to Slack!")
    elif res:
        st.warning("⚠️ API responded but Slack send failed – check the webhook.")

st.divider()

# ── Step 2: Review proposed questions ──────────────────────────────────────────

st.subheader("Step 2 · Review research questions")

data = api("get", "/state")
if data:
    st.markdown("**Current interview script:**")
    for i, q in enumerate(data["questions"], 1):
        st.markdown(f"{i}. {q}")

    # ── Add a question ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**➕ Propose additional questions**")

    with st.form("add_q"):
        new_q = st.text_area(
            "New question(s) – one per line",
            placeholder="e.g. Did you order anything else? How was it?",
        )
        submitted = st.form_submit_button("Add to script")
        if submitted and new_q.strip():
            qs = [q.strip() for q in new_q.strip().split("\n") if q.strip()]
            res = api("post", "/add-questions", json={"questions": qs})
            if res:
                st.success(f"Added {len(qs)} question(s)!")
                st.rerun()

    st.divider()

    # ── Step 3: Validate ───────────────────────────────────────────────────

    st.subheader("Step 3 · Validate the research")

    if data.get("validated"):
        st.success("✅ Research already validated.")
    else:
        if st.button("✅ Validate research & notify Slack", use_container_width=True):
            with st.spinner("Validating…"):
                res = api("post", "/validate")
            if res and res.get("validated"):
                st.success("✅ Research validated – Slack notified!")
                st.rerun()
