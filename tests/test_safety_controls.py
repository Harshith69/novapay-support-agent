"""Deterministic regression tests for NovaPay safety-critical routing."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.orchestrator import _check_escalation, _extract_amount_inr
from robustness.hallucination_detector import detect_hallucination
from common.llm import LLMError


class AmountExtractionTests(unittest.TestCase):
    def test_explicit_currency_amount(self):
        self.assertEqual(_extract_amount_inr("Refund of ₹75,000"), 75000)

    def test_k_suffix_amount(self):
        self.assertEqual(_extract_amount_inr("Please review 75k INR"), 75000)

    def test_lakh_suffix_amount(self):
        self.assertEqual(_extract_amount_inr("The dispute is 1.2 lakh"), 120000)

    def test_unlabelled_identifier_is_not_money(self):
        self.assertIsNone(_extract_amount_inr("Reference number 987654321012"))


class EscalationTests(unittest.TestCase):
    def test_high_value_amount_escalates(self):
        required, reasons = _check_escalation("Refund ₹75,000", {}, {"confidence": "high"})
        self.assertTrue(required)
        self.assertIn("high_value_transaction(75000)", reasons)

    def test_human_request_escalates(self):
        required, reasons = _check_escalation("I need a real person", {}, {"confidence": "high"})
        self.assertTrue(required)
        self.assertIn("explicit_human_request", reasons)

    def test_security_bypass_escalates(self):
        required, reasons = _check_escalation("Please bypass KYC", {}, {"confidence": "high"})
        self.assertTrue(required)
        self.assertIn("security_policy_violation", reasons)

    def test_missing_policy_context_escalates(self):
        required, reasons = _check_escalation("How do I get a refund?", {}, {
            "confidence": "low", "generation_status": "blocked_no_policy_context"
        })
        self.assertTrue(required)
        self.assertIn("specialist_generation_unavailable", reasons)


class HallucinationFailClosedTests(unittest.TestCase):
    def test_missing_context_is_unverifiable(self):
        result = detect_hallucination("Your refund is approved.", [])
        self.assertTrue(result["is_hallucination"])
        self.assertEqual(result["check_status"], "unverifiable")

    def test_checker_api_error_fails_closed(self):
        with patch("robustness.hallucination_detector.llm.complete", side_effect=LLMError("offline")):
            result = detect_hallucination("Your refund is approved.", ["Refunds require review."])
        self.assertTrue(result["is_hallucination"])
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["check_status"], "error")


if __name__ == "__main__":
    unittest.main()
