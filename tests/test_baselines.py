"""
Unit tests for numeric baselines and sample format exact matching.
"""

import unittest
import pandas as pd
from src.data_processor import load_and_preprocess_shipments, aggregate_weekly_routes
from src.baseline_calculator import compute_comparison_baselines
from src.config import SHIPMENT_RECORDS_PATH, SAMPLE_OUTPUT_PATH


class TestBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_and_preprocess_shipments(SHIPMENT_RECORDS_PATH)
        cls.weekly = aggregate_weekly_routes(cls.raw)
        cls.baselines = compute_comparison_baselines(cls.weekly)
        
        # Load sample_output_format_v2.csv safely (splits on first 7 commas to preserve unquoted text with commas)
        rows = []
        with open(SAMPLE_OUTPUT_PATH, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        headers = lines[0].split(',')
        for line in lines[1:]:
            parts = line.split(',', 7)
            rows.append(dict(zip(headers, parts)))
        cls.sample = pd.DataFrame(rows)

    def test_sample_row_1_delhi_jaipur(self):
        """Verify sample row 1: Delhi-Jaipur on 2024-11-11."""
        row = self.baselines[(self.baselines['route'] == 'Delhi-Jaipur') & (self.baselines['week_of'] == '2024-11-11')].iloc[0]
        self.assertEqual(round(row['cost_per_tonne_km'], 2), 4.17)
        self.assertIn("+35.5%", row['vs_own_history'])
        self.assertIn("+21.0%", row['vs_similar_routes'])

    def test_sample_row_2_ahmedabad_mumbai(self):
        """Verify sample row 2: Ahmedabad-Mumbai on 2025-01-20."""
        row = self.baselines[(self.baselines['route'] == 'Ahmedabad-Mumbai') & (self.baselines['week_of'] == '2025-01-20')].iloc[0]
        self.assertEqual(round(row['cost_per_tonne_km'], 2), 3.29)
        self.assertIn("+29.5%", row['vs_own_history'])
        self.assertIn("+22.5%", row['vs_similar_routes'])

    def test_sample_row_3_mumbai_pune(self):
        """Verify sample row 3: Mumbai-Pune on 2025-09-15."""
        row = self.baselines[(self.baselines['route'] == 'Mumbai-Pune') & (self.baselines['week_of'] == '2025-09-15')].iloc[0]
        self.assertEqual(round(row['cost_per_tonne_km'], 2), 3.98)
        self.assertIn("+9.2%", row['vs_own_history'])
        self.assertIn("+23.6%", row['vs_similar_routes'])

    def test_first_week_no_prior_history(self):
        """Verify week 1 has no prior look-ahead or extrapolation."""
        row = self.baselines[(self.baselines['route'] == 'Mumbai-Pune') & (self.baselines['week_of'] == '2024-01-01')].iloc[0]
        self.assertTrue(pd.isna(row['own_history_baseline']))
        self.assertIn("N/A", row['vs_own_history'])

    def test_all_routes_and_weeks_complete(self):
        """Verify all 7 routes and 104 weeks (728 route-weeks) are aggregated."""
        self.assertEqual(len(self.baselines), 728)
        self.assertEqual(len(self.baselines['route'].unique()), 7)
        self.assertEqual(len(self.baselines['week_of'].unique()), 104)


if __name__ == "__main__":
    unittest.main()
