"""NovaPay Technical Support Engineer agent."""
from __future__ import annotations

from agents.base_specialist import SpecialistAgent

SYSTEM_PROMPT = (
    "You are NovaPay's Technical Support Engineer. Help customers resolve app and "
    "account access issues. Use only the troubleshooting steps in the provided "
    "context. Give numbered step-by-step instructions. If the issue is not in the "
    "FAQ, escalate by saying 'Let me connect you with our Level 2 team' rather "
    "than guessing."
)

agent = SpecialistAgent(name="technical", system_prompt=SYSTEM_PROMPT)


def handle(query: str, triage_result: dict | None = None, extra_context: str = "") -> dict:
    return agent.handle(query, triage_result, extra_context)


if __name__ == "__main__":
    import json

    print(json.dumps(handle("I'm not receiving the OTP to log in"), indent=2, ensure_ascii=False))
