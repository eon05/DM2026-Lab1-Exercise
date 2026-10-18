"""Tools for data loading, checking, cleaning, and basic inspection."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from langchain_core.tools import tool


def make_tools(session):
    """Factory function that constructs data manipulation tools bound to the given session."""

    @tool
    def check_missing_tool(text_column: str = "text") -> Dict[str, Any]:
        """Check for missing or empty text values in the working DataFrame.

        Identifies rows where text is NaN, empty string, or whitespace-only,
        and computes missing data statistics across the current dataset.

        Args:
            text_column: Name of the column containing document text (default "text").

        Returns:
            Dict containing count and percentage of missing/empty rows,
            total row count, and a capped list of missing row indices.
        """
        if session.dataframe is None:
            return {
                "error": "No dataset loaded. Run load_dataset_tool first."
            }

        df = session.dataframe
        if text_column not in df.columns:
            return {
                "error": f"Column '{text_column}' not found in dataframe."
            }

        total_rows = len(df)
        # Catch NaNs, empty strings, and whitespace-only strings
        missing_mask = df[text_column].isna() | (df[text_column].astype(str).str.strip() == "")
        n_missing = int(missing_mask.sum())
        pct_missing = float((n_missing / total_rows * 100) if total_rows > 0 else 0.0)

        # Cap the returned missing indices list to the first 20 items to keep summary concise
        missing_indices = df.index[missing_mask].tolist()[:20]

        summary = {
            "result_id": session.next_result_id("check_missing"),
            "total_rows": total_rows,
            "n_missing": n_missing,
            "pct_missing": round(pct_missing, 4),
            "missing_indices": missing_indices
        }

        session.store_result("check_missing_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Check for duplicate documents and optionally drop them using keep=False.

        Identifies duplicate rows based on text and category_name. When drop=True,
        drops all copies of duplicate rows (keep=False) and resets downstream
        DTM matrices and label state to maintain consistency.

        Args:
            drop: If True, drop all duplicate rows from session.dataframe using keep=False.

        Returns:
            Dict containing duplicate counts, dropped row counts, and dataset status.
        """
        if session.dataframe is None:
            return {
                "error": "No dataset loaded. Run load_dataset_tool first."
            }

        df = session.dataframe
        
        # Determine subset columns to check for duplicates
        subset_cols = [c for c in ["text", "category_name"] if c in df.columns]
        if not subset_cols:
            subset_cols = list(df.columns)

        # Standard pandas duplicate count (first occurrence not marked as duplicate)
        duplicate_mask = df.duplicated(subset=subset_cols)
        n_duplicates = int(duplicate_mask.sum())

        # Duplicate mask with keep=False identifies ALL occurrences of duplicated rows
        all_duplicates_mask = df.duplicated(subset=subset_cols, keep=False)
        n_all_duplicated_rows = int(all_duplicates_mask.sum())

        rows_before = len(df)
        rows_after = rows_before
        cleared_downstream_state = False

        if drop and n_duplicates > 0:
            # Drop all copies of duplicate rows matching Master notebook behavior
            session.dataframe = df.drop_duplicates(subset=subset_cols, keep=False).reset_index(drop=True)
            rows_after = len(session.dataframe)

            # Invalidate downstream cached representations
            session.feature_matrix = None
            session.feature_names = None
            session.labels = None
            session.categories = []
            cleared_downstream_state = True

        summary = {
            "result_id": session.next_result_id("check_duplicates"),
            "n_duplicates": n_duplicates,
            "n_rows_dropped": rows_before - rows_after,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "cleared_downstream_state": cleared_downstream_state
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    return [check_missing_tool, check_duplicates_tool]
