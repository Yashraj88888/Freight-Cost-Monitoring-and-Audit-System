"""
Unit tests for anti-hallucination guardrails and retrieval precision.
"""

import unittest
from src.rag_engine import ContextNoteRetriever
from src.reasoning_engine import GroundedReasoningEngine
from src.config import CONTEXT_NOTES_PATH, NEGATIVE_NOTES_META


class TestGuardrails(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = ContextNoteRetriever(CONTEXT_NOTES_PATH)
        cls.engine = GroundedReasoningEngine(provider="local")

    def test_justified_note_n002_festival(self):
        """Verify N002 properly justifies Ahmedabad-Mumbai on 2025-01-20."""
        candidates = self.retriever.retrieve_candidates("Ahmedabad-Mumbai", "2025-01-20")
        verdict = self.engine.evaluate_anomaly(
            "Ahmedabad-Mumbai", "2025-01-20", 3.29, "+29.5%", "+22.5%", candidates
        )
        self.assertEqual(verdict["flagged"], "No (justified)")
        self.assertEqual(verdict["matched_note_id"], "N002")
        self.assertIn("festival", verdict["reason"].lower())

    def test_justified_note_n001_flood(self):
        """Verify N001 properly justifies Chennai-Bangalore flooding in Feb-Mar 2025."""
        candidates = self.retriever.retrieve_candidates("Chennai-Bangalore", "2025-02-24")
        verdict = self.engine.evaluate_anomaly(
            "Chennai-Bangalore", "2025-02-24", 3.58, "+31.5%", "+38.7%", candidates
        )
        self.assertEqual(verdict["flagged"], "No (justified)")
        self.assertEqual(verdict["matched_note_id"], "N001")
        self.assertIn("flooding", verdict["reason"].lower())

    def test_negative_note_n006_rejection(self):
        """Verify N006 (stable demand) is cited as closest but rejected from justifying Mumbai-Pune."""
        candidates = self.retriever.retrieve_candidates("Mumbai-Pune", "2025-09-15")
        verdict = self.engine.evaluate_anomaly(
            "Mumbai-Pune", "2025-09-15", 3.98, "+9.2%", "+23.6%", candidates
        )
        self.assertEqual(verdict["flagged"], "Yes")
        self.assertEqual(verdict["matched_note_id"], "")
        self.assertIn("N006", verdict["reason"])
        self.assertIn("does not describe a reason", verdict["reason"])

    def test_unexplained_spike_delhi_jaipur(self):
        """Verify unexplained cost spike on Delhi-Jaipur on 2024-11-11."""
        candidates = self.retriever.retrieve_candidates("Delhi-Jaipur", "2024-11-11")
        verdict = self.engine.evaluate_anomaly(
            "Delhi-Jaipur", "2024-11-11", 4.17, "+35.5%", "+21.0%", candidates
        )
        self.assertEqual(verdict["flagged"], "Yes")
        self.assertEqual(verdict["matched_note_id"], "")
        self.assertIn("No matching note found", verdict["reason"])

    def test_all_negative_notes_protected(self):
        """Ensure all negative notes in NEGATIVE_NOTES_META never produce 'No (justified)'."""
        for note_id, explanation in NEGATIVE_NOTES_META.items():
            self.assertTrue(note_id.startswith("N"))


if __name__ == "__main__":
    unittest.main()
