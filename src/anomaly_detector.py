"""
Anomaly detector for spotting unexpected shipping cost increases.
Flags routes where cost is rising and looks out of the ordinary compared
to own history (trailing 8 weeks) or similar routes in the same week.
"""

import pandas as pd
from typing import Tuple
from src.config import DEFAULT_OWN_THRESHOLD_PCT, DEFAULT_PEER_THRESHOLD_PCT


def detect_anomalous_route_weeks(
    df: pd.DataFrame,
    own_threshold_pct: float = DEFAULT_OWN_THRESHOLD_PCT,
    peer_threshold_pct: float = DEFAULT_PEER_THRESHOLD_PCT
) -> pd.DataFrame:
    """
    Evaluate each route-week against historical and peer thresholds.
    
    A route is considered a candidate for investigation if:
    1. Cost is rising / elevated compared to its historical rolling average (vs_own_history_pct > 0).
    2. The rise looks out of the ordinary:
       - Either it exceeds its own 8-week baseline by at least own_threshold_pct (default: 8.0%), OR
       - It exceeds similar routes in the same week by at least peer_threshold_pct (default: 20.0%).
    """
    res = df.copy()
    
    # Check conditions
    cond_own = (res['vs_own_history_pct'].fillna(0) >= own_threshold_pct)
    cond_peer = (res['vs_similar_routes_pct'].fillna(0) >= peer_threshold_pct)
    
    # Must also be rising compared to history (vs_own_history_pct > 0)
    # or strongly diverging from peers while elevated
    is_rising = (res['vs_own_history_pct'].fillna(0) > 0)
    
    # Candidate for flagging
    res['is_anomaly_candidate'] = (is_rising & (cond_own | cond_peer))
    
    return res
