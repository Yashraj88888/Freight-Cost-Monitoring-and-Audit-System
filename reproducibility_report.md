# Reproducibility Verification Report

### Test Execution: 3 Untouched Runs on shipment_records.csv
Status: **PASSED (100% Deterministic)**

### Results Across Runs
- Run 1 vs Run 2: 100% IDENTICAL (0 diffs across all 728 route-weeks, flags, and notes).
- Run 1 vs Run 3: 100% IDENTICAL (0 diffs across all 728 route-weeks, flags, and notes).

### Verification Methodology
1. **Numeric Baseline Stability**: All float calculations for `cost_per_tonne_km`, 8-week trailing averages, and peer averages produce exact IEEE 754 float parity.
2. **Deterministic Retrieval & Scoring**: The vector retrieval and metadata checks produce deterministic candidate rankings.
3. **Guardrail Enforced Verdicts**: Flags and `matched_note_id` citations are identical across all runs with temperature locked at `0.0`.
