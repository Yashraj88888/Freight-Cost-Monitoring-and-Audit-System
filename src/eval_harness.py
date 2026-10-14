"""
Lightweight Evaluation Harness for the Smart Shipping Cost Assistant.
Tests numerical baseline accuracy, positive note grounding, negative trap rejection,
and anti-hallucination compliance.
"""

import pandas as pd
from typing import Dict, Any, List
from src.data_processor import load_and_preprocess_shipments, aggregate_weekly_routes
from src.baseline_calculator import compute_comparison_baselines
from src.anomaly_detector import detect_anomalous_route_weeks
from src.rag_engine import ContextNoteRetriever
from src.reasoning_engine import GroundedReasoningEngine
from src.config import SHIPMENT_RECORDS_PATH, CONTEXT_NOTES_PATH, SAMPLE_OUTPUT_PATH


# Curated benchmark test cases testing all facets of the prompt
EVAL_TEST_CASES = [
    {
        "id": "TC01_SAMPLE_ROW1",
        "name": "Delhi-Jaipur Unexplained Cost Spike (Sample Row 1)",
        "route": "Delhi-Jaipur",
        "week_of": "2024-11-11",
        "expected_cost": 4.17,
        "expected_own_pct": "+35.5%",
        "expected_peer_pct": "+21.0%",
        "expected_flagged": "Yes",
        "expected_note": "",
        "category": "sample_contract"
    },
    {
        "id": "TC02_SAMPLE_ROW2",
        "name": "Ahmedabad-Mumbai Justified Festival Surcharge (Sample Row 2)",
        "route": "Ahmedabad-Mumbai",
        "week_of": "2025-01-20",
        "expected_cost": 3.29,
        "expected_own_pct": "+29.5%",
        "expected_peer_pct": "+22.5%",
        "expected_flagged": "No (justified)",
        "expected_note": "N002",
        "category": "sample_contract"
    },
    {
        "id": "TC03_SAMPLE_ROW3",
        "name": "Mumbai-Pune Peer Anomaly with Negative Note N006 (Sample Row 3)",
        "route": "Mumbai-Pune",
        "week_of": "2025-09-15",
        "expected_cost": 3.98,
        "expected_own_pct": "+9.2%",
        "expected_peer_pct": "+23.6%",
        "expected_flagged": "Yes",
        "expected_note": "",
        "category": "sample_contract"
    },
    {
        "id": "TC04_FLOOD_JUSTIFIED_W1",
        "name": "Chennai-Bangalore Flood Disruption Week 1 (N001)",
        "route": "Chennai-Bangalore",
        "week_of": "2025-02-24",
        "expected_cost": 3.58,
        "expected_flagged": "No (justified)",
        "expected_note": "N001",
        "category": "disruption_grounding"
    },
    {
        "id": "TC05_FLOOD_JUSTIFIED_W2",
        "name": "Chennai-Bangalore Flood Disruption Week 2 (N001 Window)",
        "route": "Chennai-Bangalore",
        "week_of": "2025-03-03",
        "expected_cost": 3.48,
        "expected_flagged": "No (justified)",
        "expected_note": "N001",
        "category": "disruption_grounding"
    },
    {
        "id": "TC06_TRAP_N004_NOT_IN_DATASET",
        "name": "Negative Trap N004 (Toll plaza not in dataset)",
        "route": "Delhi-Jaipur",
        "week_of": "2024-03-11",
        "prohibited_note": "N004",
        "category": "anti_hallucination"
    },
    {
        "id": "TC07_TRAP_N005_COSTS_NOT_AFFECTED",
        "name": "Negative Trap N005 (Maintenance where costs were not affected)",
        "route": "Mumbai-Delhi",
        "week_of": "2024-07-29",
        "prohibited_note": "N005",
        "category": "anti_hallucination"
    },
    {
        "id": "TC08_TRAP_N010_COSTS_ABSORBED",
        "name": "Negative Trap N010 (Costs absorbed without rate change)",
        "route": "Mumbai-Pune",
        "week_of": "2025-10-27",
        "prohibited_note": "N010",
        "category": "anti_hallucination"
    }
]


class EvalHarness:
    def __init__(
        self,
        provider: str = "local",
        api_key: str = None,
        model_name: str = None,
        base_url: str = None
    ):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        
    def run_eval(self) -> Dict[str, Any]:
        """Execute all test cases and compile scoring metrics."""
        # 1. Process dataset
        raw = load_and_preprocess_shipments(SHIPMENT_RECORDS_PATH)
        weekly = aggregate_weekly_routes(raw)
        baselines = compute_comparison_baselines(weekly)
        anomalies = detect_anomalous_route_weeks(baselines)
        
        retriever = ContextNoteRetriever(CONTEXT_NOTES_PATH)
        engine = GroundedReasoningEngine(
            provider=self.provider,
            api_key=self.api_key,
            model_name=self.model_name or ("qwen/qwen3.8-27b" if self.provider == "groq" else "local"),
            base_url=self.base_url,
        )
        
        results = []
        passed_count = 0
        total_count = len(EVAL_TEST_CASES)
        
        for tc in EVAL_TEST_CASES:
            tc_id = tc["id"]
            route = tc["route"]
            week_of = tc["week_of"]
            
            row_match = baselines[(baselines['route'] == route) & (baselines['week_of'] == week_of)]
            if len(row_match) == 0:
                results.append({"id": tc_id, "status": "FAIL", "reason": f"Row not found in data: {route} {week_of}"})
                continue
                
            row = row_match.iloc[0]
            cost = row['cost_per_tonne_km']
            vs_own = row['vs_own_history']
            vs_peer = row['vs_similar_routes']
            
            # Check numerical accuracy if specified
            num_passed = True
            err_msg = []
            if "expected_cost" in tc:
                if round(cost, 2) != round(tc["expected_cost"], 2):
                    num_passed = False
                    err_msg.append(f"Cost mismatch: got {cost:.2f}, expected {tc['expected_cost']}")
                    
            if "expected_own_pct" in tc:
                if tc["expected_own_pct"] not in vs_own:
                    num_passed = False
                    err_msg.append(f"vs_own mismatch: got '{vs_own}', expected contains '{tc['expected_own_pct']}'")
                    
            if "expected_peer_pct" in tc:
                if tc["expected_peer_pct"] not in vs_peer:
                    num_passed = False
                    err_msg.append(f"vs_peer mismatch: got '{vs_peer}', expected contains '{tc['expected_peer_pct']}'")
                    
            # Retrieve notes & evaluate reasoning
            candidates = retriever.retrieve_candidates(route, week_of)
            verdict_info = engine.evaluate_anomaly(
                route, week_of, cost, vs_own, vs_peer, candidates
            )
            
            flagged = verdict_info["flagged"]
            note_id = verdict_info["matched_note_id"]
            reason = verdict_info["reason"]
            
            # Check verdict correctness
            verdict_passed = True
            if "expected_flagged" in tc and flagged != tc["expected_flagged"]:
                verdict_passed = False
                err_msg.append(f"Flagged mismatch: got '{flagged}', expected '{tc['expected_flagged']}'")
                
            if "expected_note" in tc and note_id != tc["expected_note"]:
                verdict_passed = False
                err_msg.append(f"Note mismatch: got '{note_id}', expected '{tc['expected_note']}'")
                
            if "prohibited_note" in tc and note_id == tc["prohibited_note"]:
                verdict_passed = False
                err_msg.append(f"Hallucination error: prohibited note '{tc['prohibited_note']}' was cited as justification!")
                
            passed = num_passed and verdict_passed
            if passed:
                passed_count += 1
                
            results.append({
                "id": tc_id,
                "name": tc["name"],
                "category": tc["category"],
                "status": "PASS" if passed else "FAIL",
                "details": {
                    "route": route,
                    "week_of": week_of,
                    "cost": round(cost, 2),
                    "vs_own": vs_own,
                    "vs_peer": vs_peer,
                    "flagged": flagged,
                    "matched_note_id": note_id,
                    "reason": reason[:80] + "..." if len(reason) > 80 else reason,
                    "errors": err_msg
                }
            })
            
        pass_rate = (passed_count / total_count) * 100.0
        return {
            "total_tests": total_count,
            "passed_tests": passed_count,
            "pass_rate_pct": pass_rate,
            "results": results
        }
