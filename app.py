"""NovaPay Multi-Agent Support — Gradio app (HF Spaces entry point).

Two tabs:
  1. Customer Support — submit a query, see triage + grounded response +
     sources + confidence + escalation flag + metrics, and rate the answer.
  2. Operations Dashboard — live business KPIs computed from the interaction log.

The orchestrator is imported lazily (only on first query) so the Space starts
fast, and the vector store is built on first run if missing.
"""
from __future__ import annotations

import gradio as gr

from common.logging_utils import get_logger, utc_now_iso
from dashboard.metrics_tracker import log_interaction, compute_dashboard_metrics

logger = get_logger("app")

_orchestrator = None
_last_interaction: dict = {}


def _ensure_ready():
    """Lazily build the vector store and import the orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        from rag.build_vectorstore import ensure_built

        ensure_built()
        from agents.orchestrator import handle_query

        _orchestrator = handle_query
    return _orchestrator


# --------------------------------------------------------------------------- #
# Tab 1 — Customer Support
# --------------------------------------------------------------------------- #
def process_query(query: str):
    global _last_interaction
    if not query or not query.strip():
        return ("Please describe your issue.", "", "", "", "", gr.update(visible=False))

    handle_query = _ensure_ready()
    result = handle_query(query)

    tri = result.get("triage_result", {})
    badges = (
        f"**Category:** `{result.get('agent_used')}`  ·  "
        f"**Urgency:** `{tri.get('urgency', '?')}`  ·  "
        f"**Sentiment:** `{tri.get('sentiment', '?')}`  ·  "
        f"**Confidence:** `{result.get('confidence')}`"
    )
    response = result.get("response", "")
    sources = "\n".join(f"- {s}" for s in result.get("sources_used", [])) or "_none_"

    esc = result.get("escalation_required")
    esc_md = (
        f"🚨 **Escalated to human review** — {', '.join(result.get('escalation_reasons', []))}"
        if esc else "✅ Handled by AI agent"
    )

    tokens = result.get("token_counts", {})
    metrics = (
        f"Latency: {result.get('total_latency_ms', 0):.0f} ms  ·  "
        f"Tokens in/out: {tokens.get('input', 0)}/{tokens.get('output', 0)}  ·  "
        f"Hallucination: {result.get('hallucination', {}).get('severity', 'none')}"
    )

    # Stage the interaction so feedback buttons can finalise it.
    _last_interaction = {
        "timestamp": utc_now_iso(),
        "query": query,
        "category": result.get("agent_used"),
        "urgency": tri.get("urgency"),
        "agent_used": result.get("agent_used"),
        "latency_ms": result.get("total_latency_ms", 0),
        "token_count_input": tokens.get("input", 0),
        "token_count_output": tokens.get("output", 0),
        "confidence": result.get("confidence"),
        "escalation_required": esc,
        "hallucination_detected": result.get("hallucination", {}).get("is_hallucination", False),
    }
    return (response, badges, sources, esc_md, metrics, gr.update(visible=True))


def _record_feedback(rating: str, note: str = ""):
    if _last_interaction:
        log_interaction({**_last_interaction, "user_feedback": rating, "feedback_note": note})
    return gr.update(value=f"Thanks for your feedback! ({rating})", visible=True)


# --------------------------------------------------------------------------- #
# Tab 2 — Operations Dashboard
# --------------------------------------------------------------------------- #
def _refresh_metrics():
    m = compute_dashboard_metrics()
    if m.get("total_queries", 0) == 0:
        return "No interactions logged yet. Submit a query in the Support tab.", []

    summary = (
        f"### Operations Summary\n"
        f"- **Total queries:** {m['total_queries']}\n"
        f"- **Avg latency:** {m['avg_latency_ms']} ms\n"
        f"- **Escalation rate:** {m['escalation_rate_pct']}%\n"
        f"- **Hallucination rate:** {m['hallucination_rate_pct']}%\n"
        f"- **Avg confidence:** {m['avg_confidence']}\n"
        f"- **Est. cost / ticket:** ${m['estimated_cost_per_ticket_usd']}\n"
        f"- **Human minutes saved:** {m['human_minutes_saved']}\n"
    )
    table = [
        [cat, d["count"], d["avg_latency_ms"], d["escalation_rate_pct"],
         d["hallucination_rate_pct"], d["avg_confidence"]]
        for cat, d in m["by_category"].items()
    ]
    return summary, table


def build_app() -> gr.Blocks:
    with gr.Blocks(title="NovaPay Multi-Agent Support", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏦 NovaPay Multi-Agent Support System")
        gr.Markdown(
            "**Built by [Harshith Narasimhamurthy]"
            "(https://www.linkedin.com/in/harshithnarasimhamurthy69/)** "
            "| harshithnchandan@gmail.com | +91-9663918804 "
            "| Multi-agent orchestration + RAG + QLoRA fine-tuned triage",
        )

        with gr.Tab("Customer Support"):
            query_in = gr.Textbox(label="Describe your issue", lines=3,
                                  placeholder="e.g. My UPI payment failed but money was debited")
            submit = gr.Button("Submit", variant="primary")

            badges = gr.Markdown()
            response_out = gr.Textbox(label="Response", lines=8, show_copy_button=True)
            with gr.Accordion("Sources used", open=False):
                sources_out = gr.Markdown()
            escalation_out = gr.Markdown()
            with gr.Accordion("Latency & tokens", open=False):
                metrics_out = gr.Markdown()

            with gr.Row(visible=False) as feedback_row:
                up = gr.Button("👍 Helpful")
                down = gr.Button("👎 Not helpful")
            note_in = gr.Textbox(label="What was wrong? (optional)", visible=False)
            feedback_ack = gr.Markdown(visible=False)

            submit.click(
                process_query, inputs=query_in,
                outputs=[response_out, badges, sources_out, escalation_out, metrics_out, feedback_row],
            )
            up.click(lambda: _record_feedback("up"), outputs=feedback_ack)
            down.click(lambda: (gr.update(visible=True), gr.update(visible=True)),
                       outputs=[note_in, feedback_ack]).then(
                lambda note: _record_feedback("down", note), inputs=note_in, outputs=feedback_ack)

        with gr.Tab("Operations Dashboard"):
            refresh = gr.Button("🔄 Refresh Metrics", variant="primary")
            dash_summary = gr.Markdown()
            dash_table = gr.Dataframe(
                headers=["Category", "Query Count", "Avg Latency (ms)",
                         "Escalation Rate (%)", "Hallucination Rate (%)", "Avg Confidence"],
                label="Per-category breakdown", interactive=False,
            )
            refresh.click(_refresh_metrics, outputs=[dash_summary, dash_table])

    return demo


if __name__ == "__main__":
    build_app().launch()
