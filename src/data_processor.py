"""
Data processor for shipment records.
Handles date alignment to Monday-based weeks, route grouping, and weekly metric aggregation.
"""

import pandas as pd
from pathlib import Path
from typing import Union


REQUIRED_COLUMNS = [
    'shipment_id', 'shipment_date', 'origin', 'destination', 
    'route_type', 'quantity_tonnes', 'distance_km', 'freight_cost_inr'
]


def load_and_preprocess_shipments(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load raw shipment records and perform initial standardizations and validations:
    - Verifies file exists and contains all required columns.
    - Validates positive quantities and distances (guarding against division by zero).
    - Checks for non-negative freight costs.
    - Handles deduplication by shipment_id if duplicate entries exist.
    - Parses shipment_date into datetime.
    - Aligns dates to Monday-Sunday calendar weeks (week_of = Monday's date in YYYY-MM-DD).
    - Constructs canonical route identifier (Origin-Destination).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Shipment records file not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Shipment records file is empty: {file_path}")
        
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {file_path}: {missing_cols}")
    
    # Deduplicate on shipment_id if duplicate entries exist
    if df['shipment_id'].duplicated().any():
        df = df.drop_duplicates(subset=['shipment_id'], keep='first')
        
    # Validate numeric fields
    invalid_mask = (df['quantity_tonnes'] <= 0) | (df['distance_km'] <= 0) | (df['freight_cost_inr'] < 0)
    if invalid_mask.any():
        num_invalid = invalid_mask.sum()
        df = df[~invalid_mask].copy()
        if df.empty:
            raise ValueError(f"All records in {file_path} contain invalid numeric values (quantity <= 0 or distance <= 0).")
    
    # Ensure proper datetime parsing
    df['shipment_date'] = pd.to_datetime(df['shipment_date'], errors='coerce')
    if df['shipment_date'].isna().any():
        df = df.dropna(subset=['shipment_date']).copy()
        if df.empty:
            raise ValueError(f"No valid dates could be parsed from {file_path}")
    
    # Weeks run Monday to Sunday; week_of is Monday's date
    # In pandas, dt.weekday has Monday=0, Sunday=6
    df['week_of'] = df['shipment_date'].apply(
        lambda d: (d - pd.Timedelta(days=d.weekday())).strftime('%Y-%m-%d')
    )
    
    # Construct canonical route name
    df['route'] = df['origin'].str.strip() + '-' + df['destination'].str.strip()
    df['route_type'] = df['route_type'].str.strip()
    
    # Compute individual tonne-km for auditing if needed
    df['tonne_km'] = df['quantity_tonnes'] * df['distance_km']
    
    return df


def aggregate_weekly_routes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate shipments by route and week_of.
    Formula required by PDF:
    Cost per tonne per km = total freight cost / (quantity carried in tonnes x distance travelled in km)
    which equals sum(freight_cost_inr) / sum(quantity_tonnes * distance_km).
    """
    grouped = df.groupby(['route', 'route_type', 'week_of'], as_index=False).apply(
        lambda g: pd.Series({
            'total_freight_cost': g['freight_cost_inr'].sum(),
            'total_tonne_km': g['tonne_km'].sum(),
            'cost_per_tonne_km': g['freight_cost_inr'].sum() / g['tonne_km'].sum(),
            'shipment_count': len(g),
            'avg_quantity_tonnes': g['quantity_tonnes'].mean(),
            'avg_distance_km': g['distance_km'].mean()
        }),
        include_groups=False
    ).reset_index(drop=True)
    
    # Sort chronologically per route
    grouped = grouped.sort_values(['route', 'week_of']).reset_index(drop=True)
    return grouped
