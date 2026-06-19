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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Space Grotesk', 'Inter', sans-serif !important; }

/* ── FULL WIDTH — no side margins ── */
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    background: #06080f !important;
    min-height: 100vh;
}
.gradio-container > .main {
    max-width: 100% !important;
    padding: 0 24px !important;
}

/* ── Hero ── */
.hero-banner {
    position: relative;
    padding: 56px 52px 48px;
    margin: 0 -24px 16px -24px;
    background: linear-gradient(160deg, #080c1a 0%, #0f1630 50%, #0a0e1e 100%);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -120px; right: -40px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(59,130,246,0.18) 0%, transparent 65%);
    border-radius: 50%;
    filter: blur(80px);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -100px; left: 25%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(139,92,246,0.14) 0%, transparent 65%);
    border-radius: 50%;
    filter: blur(70px);
    pointer-events: none;
}
.hero-banner h1 {
    color: #ffffff !important;
    font-size: 3.6em !important;
    font-weight: 700 !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -2px;
    line-height: 1.05 !important;
    text-transform: uppercase;
    position: relative;
    z-index: 1;
}
.hero-accent {
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-banner .hero-sub {
    color: #b0bec5 !important;
    font-size: 1.15em !important;
    margin: 12px 0 0 0 !important;
    line-height: 1.7;
    max-width: 650px;
    font-weight: 400;
    position: relative;
    z-index: 1;
}
.hero-banner .hero-contact {
    color: #7e8fa6 !important;
    font-size: 0.95em !important;
    margin: 14px 0 0 0 !important;
    position: relative;
    z-index: 1;
}
.hero-banner a {
    color: #60a5fa !important;
    text-decoration: none;
    font-weight: 600;
    transition: color 0.2s;
}
.hero-banner a:hover { color: #93c5fd !important; }

/* ── Tech pills ── */
.tech-pills {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 22px;
    position: relative;
    z-index: 1;
}
.tech-pills span {
    background: rgba(255,255,255,0.05);
    color: #94a3b8;
    padding: 8px 20px;
    border-radius: 100px;
    font-size: 0.88em;
    font-weight: 500;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
    letter-spacing: 0.3px;
}
.tech-pills span:hover {
    background: rgba(59,130,246,0.1);
    border-color: rgba(59,130,246,0.25);
    color: #60a5fa;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(59,130,246,0.12);
}

/* ── Stat cards row ── */
.stat-row {
    display: flex;
    gap: 16px;
    margin-top: 28px;
    position: relative;
    z-index: 1;
}
.stat-card {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
}
.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(0,0,0,0.4);
    border-color: rgba(59,130,246,0.2);
    background: rgba(255,255,255,0.06);
}
.stat-card .stat-value {
    font-size: 2.2em;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -1px;
    line-height: 1;
}
.stat-card .stat-value .stat-unit {
    font-size: 0.4em;
    color: #4a5568;
    font-weight: 500;
}
.stat-card .stat-label {
    font-size: 0.78em;
    color: #5a6a80;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 6px;
    font-weight: 600;
}

/* ── Tab styling ── */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 1em !important;
    padding: 14px 32px !important;
    border-radius: 12px 12px 0 0 !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #5a6a80 !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-bottom: none !important;
}
.tab-nav button.selected {
    background: rgba(15,20,40,0.8) !important;
    color: #e2e8f0 !important;
    border-color: rgba(255,255,255,0.06) !important;
    border-bottom: none !important;
}

/* ── Input area — large, readable ── */
.query-input textarea {
    background: rgba(12,16,30,0.9) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    color: #e2e8f0 !important;
    font-size: 1.1em !important;
    padding: 20px 24px !important;
    line-height: 1.6 !important;
    transition: all 0.3s ease !important;
}
.query-input textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15), 0 8px 32px rgba(0,0,0,0.3) !important;
}
.query-input label span {
    font-size: 0.85em !important;
    font-weight: 600 !important;
    color: #5a6a80 !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* ── Submit button — bold ── */
.submit-btn button {
    background: linear-gradient(135deg, #3b82f6, #7c3aed) !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 18px 48px !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #fff !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 6px 24px rgba(59,130,246,0.3),
                0 0 50px rgba(124,58,237,0.08) !important;
}
.submit-btn button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 10px 36px rgba(59,130,246,0.4),
                0 0 70px rgba(124,58,237,0.12) !important;
}

/* ── Triage badges ── */
.triage-badges {
    background: rgba(12,16,30,0.9) !important;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 20px 28px;
    margin: 12px 0;
}
.triage-badges p {
    margin: 0 !important;
    color: #b0bec5 !important;
    font-size: 1.05em !important;
    line-height: 1.8 !important;
}
.triage-badges strong { color: #e2e8f0 !important; }
.triage-badges code {
    background: rgba(59,130,246,0.1) !important;
    color: #60a5fa !important;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.92em;
    border: 1px solid rgba(59,130,246,0.15);
}

/* ── Response card ── */
.response-card {
    background: rgba(12,16,30,0.9) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25) !important;
}
.response-card textarea {
    background: transparent !important;
    color: #d4dae4 !important;
    border: none !important;
    font-size: 1.05em !important;
    line-height: 1.85 !important;
    font-family: 'Inter', sans-serif !important;
}
.response-card label span {
    color: #3b82f6 !important;
    font-weight: 700 !important;
    font-size: 0.85em !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* ── Escalation status ── */
.status-escalated {
    background: rgba(12,16,30,0.9) !important;
    border-radius: 14px;
    padding: 16px 24px;
    margin: 8px 0;
    border: 1px solid rgba(255,255,255,0.06);
    font-size: 1.02em !important;
}

/* ── Accordion ── */
.gradio-accordion {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    overflow: hidden;
    margin: 6px 0 !important;
    background: rgba(12,16,30,0.8) !important;
}
.gradio-accordion .label-wrap {
    background: transparent !important;
    padding: 14px 20px !important;
}
.gradio-accordion .label-wrap span {
    color: #5a6a80 !important;
    font-weight: 600 !important;
    font-size: 0.9em !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Feedback buttons ── */
.feedback-btn button {
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    font-size: 0.95em !important;
    transition: all 0.3s ease !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    background: rgba(12,16,30,0.9) !important;
    color: #94a3b8 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.feedback-btn button:hover {
    transform: translateY(-2px) !important;
    border-color: #3b82f6 !important;
    color: #60a5fa !important;
    box-shadow: 0 8px 24px rgba(59,130,246,0.15) !important;
}

/* ── Dashboard ── */
.dash-card {
    background: rgba(12,16,30,0.9) !important;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 32px 36px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.25);
}
.dash-card h2 {
    color: #e2e8f0 !important;
    font-size: 1.4em !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}
.dash-card table { width: 100%; }
.dash-card th {
    background: rgba(59,130,246,0.06) !important;
    color: #5a6a80 !important;
    font-weight: 600 !important;
    font-size: 0.85em !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 12px 16px !important;
    border: none !important;
}
.dash-card td {
    color: #b0bec5 !important;
    padding: 12px 16px !important;
    border-bottom: 1px solid rgba(255,255,255,0.04) !important;
    font-size: 1em !important;
}
.dash-card strong { color: #e2e8f0 !important; }

.dash-refresh button {
    background: linear-gradient(135deg, #3b82f6, #7c3aed) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95em !important;
    color: #fff !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    box-shadow: 0 4px 20px rgba(59,130,246,0.25) !important;
    transition: all 0.3s ease !important;
}
.dash-refresh button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(59,130,246,0.35) !important;
}

/* ── Data table ── */
.dash-table table { border-radius: 14px !important; overflow: hidden; }
.dash-table th {
    background: rgba(59,130,246,0.06) !important;
    color: #5a6a80 !important;
    font-weight: 600 !important;
    font-size: 0.85em !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.dash-table td {
    background: rgba(12,16,30,0.8) !important;
    color: #b0bec5 !important;
    border-color: rgba(255,255,255,0.04) !important;
    font-size: 1em !important;
}

/* ── Footer ── */
.footer-section {
    text-align: center;
    padding: 28px;
    margin-top: 20px;
}
.footer-section p {
    color: #3a4560 !important;
    font-size: 0.8em !important;
    margin: 0 !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.gradio-container > .main > .wrap { animation: fadeUp 0.6s ease; }

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.processing {
    background: linear-gradient(90deg, rgba(12,16,30,0.9), rgba(59,130,246,0.06), rgba(12,16,30,0.9));
    background-size: 200% 100%;
    animation: shimmer 2s infinite;
}

/* ── Global overrides ── */
.gradio-container .prose { color: #b0bec5 !important; font-size: 1.02em !important; }
.gradio-container .block { border-color: rgba(255,255,255,0.05) !important; }
.gradio-container .label-wrap { font-size: 1em !important; }
.gradio-container .markdown p { font-size: 1.02em !important; }
.gradio-container .markdown table { font-size: 1em !important; }
.gradio-container .markdown td, .gradio-container .markdown th { padding: 10px 14px !important; }
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
        f"{cat_emoji} **Category:** `{category}`   "
        f"{urg_color} **Urgency:** `{urgency}`   "
        f"\U0001f3af **Sentiment:** `{sentiment}`   "
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

    log_interaction(_last_interaction)

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
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.violet,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Space Grotesk"),
        ),
        css=CUSTOM_CSS,
    ) as demo:

        # ── Hero banner ──
        gr.HTML("""
        <div class="hero-banner">
            <h1>NOVAPAY<br><span class="hero-accent">INTELLIGENT SUPPORT.</span></h1>
            <p class="hero-sub">
                Multi-agent AI system with intelligent routing,
                policy-grounded responses, and real-time safety guardrails.
            </p>
            <p class="hero-contact">
                Built by <a href="https://www.linkedin.com/in/harshithnarasimhamurthy69/" target="_blank">Harshith Narasimhamurthy</a>
                &nbsp;&bull;&nbsp; harshithnchandan@gmail.com &nbsp;&bull;&nbsp; +91-9663918804
            </p>
            <div class="tech-pills">
                <span>Multi-Agent Orchestration</span>
                <span>RAG + ChromaDB</span>
                <span>QLoRA Fine-Tuned Triage</span>
                <span>Llama 3.3 70B</span>
                <span>Hallucination Detection</span>
                <span>Groq Inference</span>
            </div>
            <div class="stat-row">
                <div class="stat-card">
                    <div class="stat-value">4.9<span class="stat-unit">/5.0</span></div>
                    <div class="stat-label">Judge Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">15<span class="stat-unit">/15</span></div>
                    <div class="stat-label">Adversarial Safe</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">0.74</div>
                    <div class="stat-label">BERTScore F1</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">$0</div>
                    <div class="stat-label">Infra Cost</div>
                </div>
            </div>
        </div>
        """)

        # ── Tab 1: Customer Support ──
        with gr.Tab("Customer Support", id="support"):
            gr.HTML("""
            <div style="padding: 12px 0 8px 0;">
                <p style="color: #5a6a80; font-size: 0.95em; margin: 0; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 500;">
                    Describe your banking issue &mdash; the system classifies, retrieves policy, and generates a grounded response.
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
            <div style="padding: 12px 0 8px 0;">
                <p style="color: #5a6a80; font-size: 0.95em; margin: 0; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 500;">
                    Real-time operational metrics &mdash; volume, latency, escalation rates, cost per ticket, and ROI.
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
                NovaPay Multi-Agent Support &nbsp;&bull;&nbsp;
                Llama 3.3 70B via Groq &nbsp;&bull;&nbsp;
                RAG + ChromaDB &nbsp;&bull;&nbsp;
                QLoRA Gemma 3 4B
            </p>
        </div>
        """)

    return demo


if __name__ == "__main__":
    build_app().launch()
