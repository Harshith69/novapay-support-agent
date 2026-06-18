"""NovaPay Billing Coordinator agent."""
from __future__ import annotations

from agents.base_specialist import SpecialistAgent

SYSTEM_PROMPT = (
    "You are NovaPay's Billing Coordinator. Help customers with EMI issues, "
    "subscription charges, and invoice queries. Be precise with numbers and "
    "dates. Never confirm a refund or waiver without first checking the policy "
    "context provided. If a transaction is above 50000 INR, always flag for human "
    "review."
)

agent = SpecialistAgent(name="billing", system_prompt=SYSTEM_PROMPT)


def handle(query: str, triage_result: dict | None = None, extra_context: str = "") -> dict:
    return agent.handle(query, triage_result, extra_context)


if __name__ == "__main__":
    import json

    print(json.dumps(handle("My EMI of 8500 was deducted twice this month"), indent=2, ensure_ascii=False))
