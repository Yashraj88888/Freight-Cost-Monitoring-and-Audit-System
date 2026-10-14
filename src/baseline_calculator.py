"""
Baseline calculator implementing the exact formulas specified in the PDF:
1. vs. own history: trailing 8-week rolling average cost per tonne-km strictly before current week.
2. vs. similar routes: same-week average cost per tonne-km across all other routes of the same route_type.
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_comparison_baselines(weekly_df: pd.DataFrame, window_size: int = 8) -> pd.DataFrame:
    """
    Compute numeric baselines and exact display strings for own history and similar routes.
    
    Grading specifications:
    - vs. own history: trailing window_size (8) weeks strictly prior to the current week (no look-ahead).
      If fewer than 8 prior weeks exist, use all prior weeks available without padding or extrapolation.
    - vs. similar routes: average cost per tonne-km in that same week across all other routes sharing
      the same route_type (excluding the route itself).
    """
    df = weekly_df.copy()
    
    # Initialize numeric baseline columns
    df['own_history_baseline'] = np.nan
    df['vs_own_history_pct'] = np.nan
    df['vs_own_history'] = ""
    
    df['similar_routes_baseline'] = np.nan
    df['vs_similar_routes_pct'] = np.nan
    df['vs_similar_routes'] = ""
    
    # 1. Compute vs. own history per route
    for route in df['route'].unique():
        r_indices = df[df['route'] == route].sort_values('week_of').index
        costs = df.loc[r_indices, 'cost_per_tonne_km'].values
        
        for i, idx in enumerate(r_indices):
            if i == 0:
                # First week in dataset: 0 prior weeks available
                df.loc[idx, 'own_history_baseline'] = np.nan
                df.loc[idx, 'vs_own_history_pct'] = np.nan
                df.loc[idx, 'vs_own_history'] = "N/A (first week, no prior history)"
            else:
                # Up to 8 weeks strictly prior to current week
                start_i = max(0, i - window_size)
                prior_costs = costs[start_i:i]
                rolling_avg = float(np.mean(prior_costs))
                curr_cost = costs[i]
                pct_diff = ((curr_cost - rolling_avg) / rolling_avg) * 100.0
                
                df.loc[idx, 'own_history_baseline'] = rolling_avg
                df.loc[idx, 'vs_own_history_pct'] = pct_diff
                df.loc[idx, 'vs_own_history'] = f"{pct_diff:+.1f}% vs this route's past average"
                
    # 2. Compute vs. similar routes per week and route_type
    for week in df['week_of'].unique():
        for rtype in df['route_type'].unique():
            mask = (df['week_of'] == week) & (df['route_type'] == rtype)
            group_indices = df[mask].index
            
            if len(group_indices) <= 1:
                # No peers exist in this route_type
                for idx in group_indices:
                    df.loc[idx, 'similar_routes_baseline'] = np.nan
                    df.loc[idx, 'vs_similar_routes_pct'] = np.nan
                    df.loc[idx, 'vs_similar_routes'] = "N/A (no peer routes in group)"
                continue
                
            for idx in group_indices:
                curr_cost = df.loc[idx, 'cost_per_tonne_km']
                # Exclude the route itself from peer group
                peer_costs = df.loc[group_indices[group_indices != idx], 'cost_per_tonne_km'].values
                peer_avg = float(np.mean(peer_costs))
                pct_diff = ((curr_cost - peer_avg) / peer_avg) * 100.0
                
                df.loc[idx, 'similar_routes_baseline'] = peer_avg
                df.loc[idx, 'vs_similar_routes_pct'] = pct_diff
                df.loc[idx, 'vs_similar_routes'] = f"{pct_diff:+.1f}% vs similar-length routes this week"
                
    return df
