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

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0f172a 100%) !important;
    min-height: 100vh;
}

/* ── Hero header ── */
.hero-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #0ea5e9 50%, #6366f1 100%);
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(14, 165, 233, 0.15);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner h1 {
    color: #fff !important;
    font-size: 2em !important;
    font-weight: 800 !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.5px;
}
.hero-banner p {
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.92em !important;
    margin: 0 !important;
    line-height: 1.6;
}
.hero-banner a { color: #93c5fd !important; text-decoration: none; font-weight: 600; }
.hero-banner a:hover { color: #fff !important; text-decoration: underline; }

/* ── Pill badges row ── */
.tech-pills {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 14px;
}
.tech-pills span {
    background: rgba(255,255,255,0.15);
    color: #e0f2fe;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78em;
    font-weight: 500;
    backdrop-filter: blur(4px);
    border: 1px solid rgba(255,255,255,0.1);
}

/* ── Tab styling ── */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95em !important;
    padding: 12px 24px !important;
    border-radius: 10px 10px 0 0 !important;
    transition: all 0.3s ease !important;
}
.tab-nav button.selected {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: #fff !important;
    border: none !important;
}

/* ── Input area ── */
.query-input textarea {
    background: #1e293b !important;
    border: 2px solid #334155 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 1em !important;
    padding: 16px !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
.query-input textarea:focus {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.2) !important;
}
.query-input label span {
    font-size: 0.92em !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Submit button ── */
.submit-btn button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-weight: 700 !important;
    font-size: 1em !important;
    letter-spacing: 0.3px;
    color: #fff !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(14, 165, 233, 0.3) !important;
}
.submit-btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(14, 165, 233, 0.4) !important;
}

/* ── Classification badges ── */
.triage-badges {
    background: linear-gradient(135deg, #1e293b, #0f172a) !important;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
}
.triage-badges p { margin: 0 !important; }

/* ── Response card ── */
.response-card {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}
.response-card textarea {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border: none !important;
    font-size: 0.95em !important;
    line-height: 1.7 !important;
}
.response-card label span {
    color: #0ea5e9 !important;
    font-weight: 700 !important;
    font-size: 0.85em !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Status cards (escalation) ── */
.status-escalated {
    background: linear-gradient(135deg, #1e293b, #0f172a) !important;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 4px 0;
}

/* ── Accordion styling ── */
.gradio-accordion {
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    overflow: hidden;
    margin: 4px 0 !important;
}
.gradio-accordion .label-wrap {
    background: #1e293b !important;
    padding: 10px 16px !important;
}
.gradio-accordion .label-wrap span {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.88em !important;
}

/* ── Feedback buttons ── */
.feedback-btn button {
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 0.9em !important;
    transition: all 0.3s ease !important;
    border: 1px solid #334155 !important;
    background: #1e293b !important;
    color: #e2e8f0 !important;
}
.feedback-btn button:hover {
    transform: translateY(-1px) !important;
    border-color: #0ea5e9 !important;
    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.2) !important;
}

/* ── Dashboard cards ── */
.dash-card {
    background: linear-gradient(135deg, #1e293b, #0f172a) !important;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 24px;
}
.dash-card h3 { color: #0ea5e9 !important; }
.dash-card li { color: #cbd5e1 !important; }
.dash-card strong { color: #e2e8f0 !important; }

.dash-refresh button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    color: #fff !important;
    box-shadow: 0 4px 20px rgba(14, 165, 233, 0.3) !important;
}

/* ── Data table ── */
.dash-table table {
    border-radius: 10px !important;
    overflow: hidden;
}
.dash-table th {
    background: #1e3a5f !important;
    color: #e0f2fe !important;
    font-weight: 600 !important;
    font-size: 0.85em !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.dash-table td {
    background: #1e293b !important;
    color: #cbd5e1 !important;
    border-color: #334155 !important;
}

/* ── Footer ── */
.footer-section {
    text-align: center;
    padding: 20px;
    margin-top: 12px;
}
.footer-section p {
    color: #475569 !important;
    font-size: 0.8em !important;
    margin: 0 !important;
}

/* ── Animations ── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.gradio-container > .main > .wrap { animation: fadeIn 0.5s ease-out; }

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.processing {
    background: linear-gradient(90deg, #1e293b, #334155, #1e293b);
    background-size: 200% 100%;
    animation: shimmer 2s infinite;
}
"""


def _ensure_ready():
    global _orchestrator
    if _orchestrator is None:
        from rag.build_vectorstore import ensure_built
        ensure_built()
        from agents.orchestrator import handle_query
        _orchestrator = handle_query
    return _orchestrator


def process_query(query: str):
    global _last_interaction
    if not query or not query.strip():
        return ("Please describe your issue.", "", "", "", "", gr.update(visible=False))

    handle_query = _ensure_ready()
    result = handle_query(query)

    tri = result.get("triage_result", {})
    category = result.get("agent_used", "?")
    urgency = tri.get("urgency", "?")
    sentiment = tri.get("sentiment", "?")
    confidence = result.get("confidence", "?")

    cat_emoji = {"refund": "\U0001f4b8", "technical": "⚙️", "billing": "\U0001f4cb"}.get(category, "\U0001f4ac")
    urg_color = {"high": "\U0001f534", "medium": "\U0001f7e1", "low": "\U0001f7e2"}.get(urgency, "⚪")

    badges = (
        f"{cat_emoji} **Category:** `{category}`&nbsp;&nbsp;&nbsp;"
        f"{urg_color} **Urgency:** `{urgency}`&nbsp;&nbsp;&nbsp;"
        f"\U0001f3af **Sentiment:** `{sentiment}`&nbsp;&nbsp;&nbsp;"
        f"\U0001f4ca **Confidence:** `{confidence}`"
    )

    response = result.get("response", "")
    src_list = result.get("sources_used", [])
    sources = "\n".join(f"\U0001f4c4 `{s}`" for s in src_list) if src_list else "— _No external sources used_"

    esc = result.get("escalation_required")
    reasons = result.get("escalation_reasons", [])
    if esc:
        esc_md = f"\U0001f6a8 **Escalated to Human Review**\n\n> Reason: _{', '.join(reasons)}_"
    else:
        esc_md = "✅ **Resolved by AI Agent** — No escalation needed"

    tokens = result.get("token_counts", {})
    latency = result.get("total_latency_ms", 0)
    hall = result.get("hallucination", {})
    metrics = (
        f"| Metric | Value |\n|---|---|\n"
        f"| ⏱️ Latency | **{latency:.0f} ms** |\n"
        f"| \U0001f4e5 Input tokens | {tokens.get('input', 0)} |\n"
        f"| \U0001f4e4 Output tokens | {tokens.get('output', 0)} |\n"
        f"| \U0001f50d Hallucination check | `{hall.get('severity', 'none')}` |\n"
        f"| \U0001f916 Triage source | `{tri.get('source', '?')}` |"
    )

    _last_interaction = {
        "timestamp": utc_now_iso(),
        "query": query,
        "category": category,
        "urgency": urgency,
        "agent_used": category,
        "latency_ms": latency,
        "token_count_input": tokens.get("input", 0),
        "token_count_output": tokens.get("output", 0),
        "confidence": confidence,
        "escalation_required": esc,
        "hallucination_detected": hall.get("is_hallucination", False),
    }
    return (response, badges, sources, esc_md, metrics, gr.update(visible=True))


def _record_feedback(rating: str, note: str = ""):
    if _last_interaction:
        log_interaction({**_last_interaction, "user_feedback": rating, "feedback_note": note})
    emoji = "\U0001f44d" if rating == "up" else "\U0001f44e"
    return gr.update(value=f"{emoji} Thanks for your feedback!", visible=True)


def _refresh_metrics():
    m = compute_dashboard_metrics()
    if m.get("total_queries", 0) == 0:
        return "No interactions logged yet. Submit a query in the **Customer Support** tab first.", []

    summary = (
        f"## \U0001f4ca Operations Summary\n\n"
        f"| KPI | Value |\n|---|---|\n"
        f"| \U0001f4e8 Total queries | **{m['total_queries']}** |\n"
        f"| ⏱️ Avg latency | **{m['avg_latency_ms']} ms** |\n"
        f"| \U0001f6a8 Escalation rate | **{m['escalation_rate_pct']}%** |\n"
        f"| \U0001f50d Hallucination rate | **{m['hallucination_rate_pct']}%** |\n"
        f"| \U0001f3af Avg confidence | **{m['avg_confidence']}** |\n"
        f"| \U0001f4b0 Cost / ticket | **${m['estimated_cost_per_ticket_usd']}** |\n"
        f"| ⏳ Human minutes saved | **{m['human_minutes_saved']}** |"
    )
    table = [
        [cat, d["count"], d["avg_latency_ms"], d["escalation_rate_pct"],
         d["hallucination_rate_pct"], d["avg_confidence"]]
        for cat, d in m["by_category"].items()
    ]
    return summary, table


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="NovaPay Multi-Agent Support",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.sky,
            secondary_hue=gr.themes.colors.indigo,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CUSTOM_CSS,
    ) as demo:

        # ── Hero banner ──
        gr.HTML("""
        <div class="hero-banner">
            <h1>NovaPay Multi-Agent Support</h1>
            <p>
                AI-powered customer support with intelligent query routing,
                policy-grounded responses, and real-time safety guardrails.
            </p>
            <p style="margin-top: 10px !important;">
                Built by <a href="https://www.linkedin.com/in/harshithnarasimhamurthy69/" target="_blank">Harshith Narasimhamurthy</a>
                &nbsp;|&nbsp; harshithnchandan@gmail.com &nbsp;|&nbsp; +91-9663918804
            </p>
            <div class="tech-pills">
                <span>Multi-Agent Orchestration</span>
                <span>RAG + ChromaDB</span>
                <span>QLoRA Fine-Tuned Triage</span>
                <span>Llama 3.3 70B</span>
                <span>Hallucination Detection</span>
                <span>Groq Free Tier</span>
            </div>
        </div>
        """)

        # ── Tab 1: Customer Support ──
        with gr.Tab("Customer Support", id="support"):
            gr.HTML("""
            <div style="padding: 8px 0 4px 0;">
                <p style="color: #94a3b8; font-size: 0.88em; margin: 0;">
                    Describe your banking issue below. The system will classify your query,
                    retrieve relevant policy documents, and generate a grounded response.
                </p>
            </div>
            """)

            with gr.Group():
                query_in = gr.Textbox(
                    label="YOUR ISSUE",
                    lines=3,
                    placeholder="e.g. My UPI payment failed but Rs 2000 was debited from my account...",
                    elem_classes=["query-input"],
                )
                submit = gr.Button(
                    "Submit Query",
                    variant="primary",
                    elem_classes=["submit-btn"],
                    size="lg",
                )

            badges = gr.Markdown(elem_classes=["triage-badges"])

            response_out = gr.Textbox(
                label="AGENT RESPONSE",
                lines=8,
                show_copy_button=True,
                elem_classes=["response-card"],
            )

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("Sources & References", open=False):
                        sources_out = gr.Markdown()
                with gr.Column(scale=1):
                    with gr.Accordion("Performance Metrics", open=False):
                        metrics_out = gr.Markdown()

            escalation_out = gr.Markdown(elem_classes=["status-escalated"])

            with gr.Row(visible=False, elem_classes=["feedback-btn"]) as feedback_row:
                up = gr.Button("Helpful", size="sm")
                down = gr.Button("Not helpful", size="sm")
            note_in = gr.Textbox(
                label="What could be improved? (optional)",
                visible=False,
                elem_classes=["query-input"],
            )
            feedback_ack = gr.Markdown(visible=False)

            submit.click(
                process_query,
                inputs=query_in,
                outputs=[response_out, badges, sources_out, escalation_out, metrics_out, feedback_row],
            )
            up.click(lambda: _record_feedback("up"), outputs=feedback_ack)
            down.click(
                lambda: (gr.update(visible=True), gr.update(visible=True)),
                outputs=[note_in, feedback_ack],
            ).then(
                lambda note: _record_feedback("down", note),
                inputs=note_in,
                outputs=feedback_ack,
            )

        # ── Tab 2: Operations Dashboard ──
        with gr.Tab("Operations Dashboard", id="dashboard"):
            gr.HTML("""
            <div style="padding: 8px 0 4px 0;">
                <p style="color: #94a3b8; font-size: 0.88em; margin: 0;">
                    Real-time operational metrics computed from the interaction log.
                    Tracks volume, latency, escalation rates, cost per ticket, and ROI.
                </p>
            </div>
            """)

            refresh = gr.Button(
                "Refresh Metrics",
                variant="primary",
                elem_classes=["dash-refresh"],
                size="lg",
            )
            dash_summary = gr.Markdown(elem_classes=["dash-card"])
            dash_table = gr.Dataframe(
                headers=[
                    "Category", "Queries", "Avg Latency (ms)",
                    "Escalation %", "Hallucination %", "Confidence",
                ],
                label="Per-Category Breakdown",
                interactive=False,
                elem_classes=["dash-table"],
            )
            refresh.click(_refresh_metrics, outputs=[dash_summary, dash_table])

        # ── Footer ──
        gr.HTML("""
        <div class="footer-section">
            <p>
                NovaPay Multi-Agent Support System &nbsp;|&nbsp;
                Powered by Llama 3.3 70B via Groq &nbsp;|&nbsp;
                RAG with ChromaDB + MiniLM &nbsp;|&nbsp;
                QLoRA Fine-Tuned Gemma 3 4B
            </p>
        </div>
        """)

    return demo


if __name__ == "__main__":
    build_app().launch()
