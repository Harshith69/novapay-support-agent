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

:root {
    --bg-primary: #050508;
    --bg-secondary: #0a0a10;
    --bg-card: rgba(15, 15, 25, 0.7);
    --bg-glass: rgba(255, 255, 255, 0.03);
    --border-glass: rgba(255, 255, 255, 0.06);
    --accent-blue: #3b82f6;
    --accent-violet: #8b5cf6;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #475569;
    --glow-blue: rgba(59, 130, 246, 0.15);
    --glow-violet: rgba(139, 92, 246, 0.12);
}

* { font-family: 'Space Grotesk', 'Inter', sans-serif !important; }

.gradio-container {
    max-width: 1060px !important;
    margin: 0 auto !important;
    background: var(--bg-primary) !important;
    min-height: 100vh;
}

/* ── 3D Hero Banner ── */
.hero-banner {
    position: relative;
    padding: 48px 44px 40px;
    margin-bottom: 12px;
    border-radius: 24px;
    background: linear-gradient(160deg, #0c0c18 0%, #111128 40%, #0a0a1a 100%);
    border: 1px solid var(--border-glass);
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -100px; right: -60px;
    width: 350px; height: 350px;
    background: radial-gradient(circle, var(--glow-blue) 0%, transparent 70%);
    border-radius: 50%;
    filter: blur(60px);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 280px; height: 280px;
    background: radial-gradient(circle, var(--glow-violet) 0%, transparent 70%);
    border-radius: 50%;
    filter: blur(50px);
    pointer-events: none;
}
.hero-banner h1 {
    color: #fff !important;
    font-size: 2.8em !important;
    font-weight: 700 !important;
    margin: 0 0 4px 0 !important;
    letter-spacing: -1.5px;
    line-height: 1.1 !important;
    text-transform: uppercase;
}
.hero-banner .hero-accent {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-violet));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-banner p {
    color: var(--text-secondary) !important;
    font-size: 0.95em !important;
    margin: 0 !important;
    line-height: 1.7;
    max-width: 580px;
    font-weight: 400;
}
.hero-banner a {
    color: var(--accent-blue) !important;
    text-decoration: none;
    font-weight: 600;
    transition: color 0.2s;
}
.hero-banner a:hover { color: var(--accent-cyan) !important; }

/* ── 3D Floating pills ── */
.tech-pills {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 20px;
    position: relative;
    z-index: 1;
}
.tech-pills span {
    background: var(--bg-glass);
    color: var(--text-secondary);
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 0.78em;
    font-weight: 500;
    border: 1px solid var(--border-glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    letter-spacing: 0.3px;
}
.tech-pills span:hover {
    background: rgba(59, 130, 246, 0.08);
    border-color: rgba(59, 130, 246, 0.2);
    color: var(--accent-blue);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.1);
}

/* ── 3D Stat cards (hero row) ── */
.stat-row {
    display: flex;
    gap: 12px;
    margin-top: 24px;
    position: relative;
    z-index: 1;
}
.stat-card {
    flex: 1;
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 16px 20px;
    backdrop-filter: blur(12px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.stat-card:hover {
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    border-color: rgba(59, 130, 246, 0.15);
}
.stat-card .stat-value {
    font-size: 1.5em;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.5px;
}
.stat-card .stat-label {
    font-size: 0.72em;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* ── Tab styling ── */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.88em !important;
    padding: 12px 28px !important;
    border-radius: 12px 12px 0 0 !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.3px;
    text-transform: uppercase;
    color: var(--text-muted) !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-bottom: none !important;
}
.tab-nav button.selected {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-glass) !important;
    border-bottom: none !important;
}

/* ── Glassmorphism Input ── */
.query-input textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 16px !important;
    color: var(--text-primary) !important;
    font-size: 0.95em !important;
    padding: 18px 20px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    backdrop-filter: blur(8px);
}
.query-input textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px var(--glow-blue), 0 8px 32px rgba(0,0,0,0.3) !important;
}
.query-input label span {
    font-size: 0.75em !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* ── 3D Submit button ── */
.submit-btn button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-violet)) !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 40px !important;
    font-weight: 700 !important;
    font-size: 0.92em !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #fff !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.25),
                0 0 40px rgba(139, 92, 246, 0.08) !important;
    position: relative;
    overflow: hidden;
}
.submit-btn button::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, transparent 40%, rgba(255,255,255,0.1) 50%, transparent 60%);
    transform: translateX(-100%);
    transition: transform 0.6s ease;
}
.submit-btn button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 8px 32px rgba(59, 130, 246, 0.35),
                0 0 60px rgba(139, 92, 246, 0.12) !important;
}
.submit-btn button:hover::after { transform: translateX(100%); }

/* ── Triage badges — glass card ── */
.triage-badges {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 18px 24px;
    margin: 10px 0;
    backdrop-filter: blur(12px);
}
.triage-badges p { margin: 0 !important; color: var(--text-secondary) !important; }
.triage-badges strong { color: var(--text-primary) !important; }
.triage-badges code {
    background: rgba(59, 130, 246, 0.08) !important;
    color: var(--accent-blue) !important;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.88em;
    border: 1px solid rgba(59, 130, 246, 0.12);
}

/* ── Response card — depth effect ── */
.response-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2) !important;
}
.response-card textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
    border: none !important;
    font-size: 0.93em !important;
    line-height: 1.8 !important;
    font-family: 'Inter', sans-serif !important;
}
.response-card label span {
    color: var(--accent-blue) !important;
    font-weight: 700 !important;
    font-size: 0.75em !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* ── Status cards ── */
.status-escalated {
    background: var(--bg-card) !important;
    border-radius: 14px;
    padding: 14px 20px;
    margin: 6px 0;
    border: 1px solid var(--border-glass);
}

/* ── Accordion — depth ── */
.gradio-accordion {
    border: 1px solid var(--border-glass) !important;
    border-radius: 14px !important;
    overflow: hidden;
    margin: 6px 0 !important;
    background: var(--bg-card) !important;
}
.gradio-accordion .label-wrap {
    background: transparent !important;
    padding: 12px 18px !important;
}
.gradio-accordion .label-wrap span {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 0.82em !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Feedback — floating buttons ── */
.feedback-btn button {
    border-radius: 12px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    font-size: 0.85em !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border: 1px solid var(--border-glass) !important;
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.feedback-btn button:hover {
    transform: translateY(-2px) !important;
    border-color: var(--accent-blue) !important;
    color: var(--accent-blue) !important;
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.12) !important;
}

/* ── Dashboard — 3D floating cards ── */
.dash-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glass);
    border-radius: 20px;
    padding: 28px 32px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.2);
}
.dash-card h2 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}
.dash-card table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    width: 100%;
}
.dash-card th {
    background: rgba(59, 130, 246, 0.06) !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.75em !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 10px 14px !important;
    border: none !important;
}
.dash-card td {
    color: var(--text-secondary) !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid var(--border-glass) !important;
}
.dash-card strong {
    color: var(--text-primary) !important;
}

.dash-refresh button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-violet)) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    color: #fff !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 0.85em !important;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.25) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.dash-refresh button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(59, 130, 246, 0.35) !important;
}

/* ── Data table — depth ── */
.dash-table table {
    border-radius: 14px !important;
    overflow: hidden;
}
.dash-table th {
    background: rgba(59, 130, 246, 0.06) !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.78em !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.dash-table td {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border-color: var(--border-glass) !important;
}

/* ── Footer — minimal ── */
.footer-section {
    text-align: center;
    padding: 24px;
    margin-top: 16px;
}
.footer-section p {
    color: var(--text-muted) !important;
    font-size: 0.72em !important;
    margin: 0 !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.gradio-container > .main > .wrap { animation: fadeUp 0.6s cubic-bezier(0.4, 0, 0.2, 1); }

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.processing {
    background: linear-gradient(90deg, var(--bg-card), rgba(59,130,246,0.05), var(--bg-card));
    background-size: 200% 100%;
    animation: shimmer 2s infinite;
}

/* ── Global overrides for dark theme consistency ── */
.gradio-container .prose { color: var(--text-secondary) !important; }
.gradio-container .block { border-color: var(--border-glass) !important; }
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
        f"{cat_emoji} **Category:** `{category}`   "
        f"{urg_color} **Urgency:** `{urgency}`   "
        f"\U0001f3af **Sentiment:** `{sentiment}`   "
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

        # ── Hero banner — 3D inspired ──
        gr.HTML("""
        <div class="hero-banner">
            <h1>NOVAPAY<br><span class="hero-accent">INTELLIGENT SUPPORT.</span></h1>
            <p>
                Multi-agent AI system with intelligent routing,
                policy-grounded responses, and real-time safety guardrails.
            </p>
            <p style="margin-top: 12px !important; font-size: 0.82em !important;">
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
                    <div class="stat-value">4.9<span style="font-size:0.5em;color:#64748b">/5.0</span></div>
                    <div class="stat-label">Judge Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">15<span style="font-size:0.5em;color:#64748b">/15</span></div>
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
            <div style="padding: 10px 0 6px 0;">
                <p style="color: #475569; font-size: 0.82em; margin: 0; text-transform: uppercase; letter-spacing: 1px; font-weight: 500;">
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
            <div style="padding: 10px 0 6px 0;">
                <p style="color: #475569; font-size: 0.82em; margin: 0; text-transform: uppercase; letter-spacing: 1px; font-weight: 500;">
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
