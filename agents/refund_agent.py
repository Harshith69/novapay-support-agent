"""NovaPay Refund Specialist agent."""
from __future__ import annotations

from agents.base_specialist import SpecialistAgent

SYSTEM_PROMPT = (
    "You are NovaPay's Refund Specialist. You help customers with disputed "
    "transactions and refund requests. Answer ONLY based on the provided NovaPay "
    "policy context. If the answer is not in the context, say 'I need to check "
    "this with our team' rather than inventing a policy. Always acknowledge the "
    "customer's frustration before explaining the process."
)

agent = SpecialistAgent(name="refund", system_prompt=SYSTEM_PROMPT)


def handle(query: str, triage_result: dict | None = None, extra_context: str = "") -> dict:
    return agent.handle(query, triage_result, extra_context)


if __name__ == "__main__":
    import json

    print(json.dumps(handle("My UPI payment failed but money was debited 4 days ago"), indent=2, ensure_ascii=False))
