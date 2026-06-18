"""Shared infrastructure for the NovaPay multi-agent support system.

Centralises configuration, logging, the resilient Anthropic client and
small parsing helpers so every module (data gen, agents, evaluation,
robustness, dashboard, app) speaks the same language.
"""

from common.config import settings

__all__ = ["settings"]
