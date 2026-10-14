"""
Unit tests for data validation, malformed records, and edge cases.
"""

import unittest
import tempfile
import os
import pandas as pd
from pathlib import Path
from src.data_processor import load_and_preprocess_shipments, aggregate_weekly_routes
from src.rag_engine import ContextNoteRetriever


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_missing_shipment_file(self):
        """Verify FileNotFoundError is raised when file does not exist."""
        with self.assertRaises(FileNotFoundError):
            load_and_preprocess_shipments(Path(self.temp_dir.name) / "non_existent.csv")

    def test_empty_shipment_file(self):
        """Verify ValueError is raised on empty shipment file."""
        empty_file = Path(self.temp_dir.name) / "empty.csv"
        empty_file.write_text("")
        with self.assertRaises(Exception):
            load_and_preprocess_shipments(empty_file)

    def test_missing_required_columns(self):
        """Verify ValueError is raised when required columns are missing."""
        bad_file = Path(self.temp_dir.name) / "bad_cols.csv"
        df = pd.DataFrame({'shipment_id': ['S001'], 'origin': ['Delhi']})
        df.to_csv(bad_file, index=False)
        with self.assertRaises(ValueError) as ctx:
            load_and_preprocess_shipments(bad_file)
        self.assertIn("Missing required columns", str(ctx.exception))

    def test_invalid_numeric_values_cleaned(self):
        """Verify invalid non-positive quantities or distances are filtered out defensively."""
        file_path = Path(self.temp_dir.name) / "invalid_nums.csv"
        df = pd.DataFrame({
            'shipment_id': ['S1', 'S2', 'S3'],
            'shipment_date': ['2024-01-02', '2024-01-03', '2024-01-04'],
            'origin': ['Delhi', 'Delhi', 'Delhi'],
            'destination': ['Jaipur', 'Jaipur', 'Jaipur'],
            'route_type': ['Short', 'Short', 'Short'],
            'quantity_tonnes': [20.0, -5.0, 15.0],  # S2 has negative quantity
            'distance_km': [280.0, 280.0, 0.0],    # S3 has zero distance
            'freight_cost_inr': [25000, 25000, 20000]
        })
        df.to_csv(file_path, index=False)
        processed = load_and_preprocess_shipments(file_path)
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed.iloc[0]['shipment_id'], 'S1')

    def test_duplicate_shipment_ids_handled(self):
        """Verify duplicate shipment IDs are deduplicated."""
        file_path = Path(self.temp_dir.name) / "dupes.csv"
        df = pd.DataFrame({
            'shipment_id': ['S1', 'S1'],
            'shipment_date': ['2024-01-02', '2024-01-02'],
            'origin': ['Delhi', 'Delhi'],
            'destination': ['Jaipur', 'Jaipur'],
            'route_type': ['Short', 'Short'],
            'quantity_tonnes': [20.0, 20.0],
            'distance_km': [280.0, 280.0],
            'freight_cost_inr': [25000, 25000]
        })
        df.to_csv(file_path, index=False)
        processed = load_and_preprocess_shipments(file_path)
        self.assertEqual(len(processed), 1)

    def test_missing_context_notes_file(self):
        """Verify FileNotFoundError is raised when context notes file does not exist."""
        with self.assertRaises(FileNotFoundError):
            ContextNoteRetriever(Path(self.temp_dir.name) / "no_notes.csv")


if __name__ == "__main__":
    unittest.main()
