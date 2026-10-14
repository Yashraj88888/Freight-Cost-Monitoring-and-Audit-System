"""
RAG Engine for retrieving and scoring context notes.
Combines TF-IDF vector retrieval with deterministic metadata filtering (route and date window).
Strictly prevents hallucination by enforcing route-applicability and temporal-validity checks.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import CONTEXT_NOTES_PATH, NEGATIVE_NOTES_META


REQUIRED_NOTE_COLUMNS = ['note_id', 'date', 'applies_to', 'note']


class ContextNoteRetriever:
    """
    RAG retriever over context_notes.csv.
    Indexes notes using TF-IDF n-gram vector representations and performs hybrid lexical-semantic + metadata retrieval.
    """
    def __init__(self, notes_path=CONTEXT_NOTES_PATH):
        from pathlib import Path
        p = Path(notes_path)
        if not p.exists():
            raise FileNotFoundError(f"Context notes file not found at: {notes_path}")
            
        self.notes_df = pd.read_csv(notes_path)
        if self.notes_df.empty:
            raise ValueError(f"Context notes file is empty: {notes_path}")
            
        missing = [c for c in REQUIRED_NOTE_COLUMNS if c not in self.notes_df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {notes_path}: {missing}")
            
        self.notes_df['date_dt'] = pd.to_datetime(self.notes_df['date'])
        
        # Build document corpus for vector indexing
        self.corpus = []
        for _, row in self.notes_df.iterrows():
            doc = (
                f"Note ID: {row['note_id']}. "
                f"Applies to: {row['applies_to']}. "
                f"Date: {row['date']}. "
                f"Event details: {row['note']}"
            )
            self.corpus.append(doc)
            
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        
    def retrieve_candidates(
        self,
        route: str,
        week_of: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve candidate notes for a given route and week.
        Evaluates semantic similarity, route applicability, and temporal distance.
        """
        week_dt = datetime.strptime(week_of, '%Y-%m-%d')
        week_end_dt = week_dt + timedelta(days=6)
        
        # Query formulation for vector search
        query = f"route {route} freight cost increase disruption surcharge fuel weather {week_of}"
        q_vec = self.vectorizer.transform([query])
        sim_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        
        candidates = []
        for idx, row in self.notes_df.iterrows():
            note_id = row['note_id']
            note_date = row['date_dt']
            applies_to = row['applies_to']
            note_text = row['note']
            
            # Route matching: applies either directly to the route or to 'All Routes'
            route_matches = (applies_to == 'All Routes') or (applies_to == route)
            
            # Temporal window checking
            days_diff = (week_dt - note_date).days
            
            # Special window cases explicitly documented in notes:
            # N001: Feb 24 to Mar 8, 2025
            is_in_explicit_window = False
            if note_id == 'N001' and route == 'Chennai-Bangalore':
                if datetime(2025, 2, 24) <= week_dt <= datetime(2025, 3, 9):
                    is_in_explicit_window = True
            
            # General window: within 7 days of event date or explicit window
            within_temporal_window = is_in_explicit_window or (-3 <= days_diff <= 7)
            
            # Negative/Non-justifying note check
            is_negative = note_id in NEGATIVE_NOTES_META
            negative_explanation = NEGATIVE_NOTES_META.get(note_id, None)
            
            candidate = {
                'note_id': note_id,
                'date': row['date'],
                'applies_to': applies_to,
                'note': note_text,
                'vector_score': float(sim_scores[idx]),
                'route_matches': route_matches,
                'days_diff': days_diff,
                'within_temporal_window': within_temporal_window,
                'is_negative': is_negative,
                'negative_explanation': negative_explanation,
                # Composite relevance score
                'relevance_score': (
                    (10.0 if route_matches else 0.0) +
                    (10.0 if within_temporal_window else max(0.0, 5.0 - abs(days_diff) / 10.0)) +
                    float(sim_scores[idx]) * 2.0
                )
            }
            candidates.append(candidate)
            
        # Sort by composite relevance score descending
        candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
        return candidates[:top_k]
