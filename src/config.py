"""
Configuration and constants for FreightTiger Smart Shipping Cost Assistant.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SHIPMENT_RECORDS_PATH = BASE_DIR / "shipment_records.csv"
CONTEXT_NOTES_PATH = BASE_DIR / "context_notes.csv"
SAMPLE_OUTPUT_PATH = BASE_DIR / "sample_output_format_v2.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "output.csv"
ALL_WEEKS_OUTPUT_PATH = BASE_DIR / "output_all_weeks.csv"

# Automatically load local .env file into environment if present
env_file = BASE_DIR / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

# Core Baseline Settings
ROLLING_WINDOW_WEEKS = 8  # Strictly prior weeks (no look-ahead)

# Anomaly Detection Thresholds
# A route-week is flagged when cost is rising and exceeds historical or peer bounds
DEFAULT_OWN_THRESHOLD_PCT = 8.0    # % above trailing 8-week rolling average
DEFAULT_PEER_THRESHOLD_PCT = 20.0  # % above same-week peer group average

# Model Pricing (per 1,000,000 tokens)
# Used for published rate cost accounting
MODEL_PRICING = {
    "local": {
        "input_per_million": 0.0,
        "output_per_million": 0.0,
        "name": "Local Grounded Engine (Free / Offline)"
    },
    "qwen/qwen3.8-27b": {
        "input_per_million": 0.11,
        "output_per_million": 0.22,
        "name": "Groq Cloud LPU (qwen/qwen3.8-27b)"
    }
}

# Notes that explicitly state costs were NOT affected or do NOT justify rate hikes
# Guardrail knowledge base to prevent hallucinations
NEGATIVE_NOTES_META = {
    "N004": "Toll plaza commissioned on highway not in this dataset; does not affect these routes.",
    "N005": "Maintenance caused minor delays; costs were explicitly not significantly affected.",
    "N006": "Logistics report notes stable freight capacity demand with no major disruptions.",
    "N007": "Improved road conditions after resurfacing; does not explain a cost increase.",
    "N008": "Corridor reported no significant disruptions; freight movement remained normal.",
    "N009": "Route confirmed returned to normal conditions after flood repairs completed.",
    "N010": "Fleet mandate compliance costs were absorbed by transporters without a rate change."
}
