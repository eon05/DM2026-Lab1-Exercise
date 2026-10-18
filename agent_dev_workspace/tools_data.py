"""Tools for data loading, checking, cleaning, and basic inspection."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
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

        # Determine subset columns to check for duplicates (avoid unigrams list columns)
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

    @tool
    def sample_data_tool(n: int, random_state: int = 42) -> Dict[str, Any]:
        """Subsample the working DataFrame and reset downstream DTM state.

        Draws n random rows from session.dataframe using pandas DataFrame.sample
        with a specified random seed. Captures the original indices of sampled rows
        before resetting the DataFrame index, and invalidates downstream DTM state
        to prevent matrix shape mismatches.

        Args:
            n: Number of documents to sample.
            random_state: Seed for reproducible random sampling (default 42).

        Returns:
            Dict containing sample count, original sampled row indices, and status.
        """
        if session.dataframe is None:
            return {
                "error": "No dataset loaded. Run load_dataset_tool first."
            }

        df = session.dataframe
        total_rows_before = len(df)

        if n <= 0 or n > total_rows_before:
            return {
                "error": f"Sample size n={n} must be between 1 and total rows ({total_rows_before})."
            }

        sampled_df = df.sample(n=n, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()[:20]

        session.dataframe = sampled_df.reset_index(drop=True)

        # Invalidate downstream cached DTM representations to remain consistent
        session.feature_matrix = None
        session.feature_names = None
        session.labels = None
        session.categories = []

        summary = {
            "result_id": session.next_result_id("sample_data"),
            "n_samples": n,
            "random_state": random_state,
            "sampled_indices": sampled_indices,
            "total_rows_before": total_rows_before,
            "total_rows_after": len(session.dataframe),
            "cleared_downstream_state": True
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def inspect_data_tool(n: int = 5) -> Dict[str, Any]:
        """Inspect a sample preview of rows and dataset dimensions from session.dataframe.

        Previews up to n rows (capped at max 20) with truncated text and category names,
        omitting heavy columns like unigrams lists. Also reports the total row/column
        dimensions and column names of the working dataset.

        Args:
            n: Number of rows to inspect (default 5, capped between 1 and 20).

        Returns:
            Dict containing preview rows, dataset shape, column list, and status notes.
        """
        if session.dataframe is None:
            return {
                "error": "No dataset loaded. Run load_dataset_tool first."
            }

        df = session.dataframe
        total_rows = len(df)

        note = None
        actual_n = n
        if n > 20:
            actual_n = 20
            note = f"Requested n={n} exceeds maximum allowed (20). Capped n to 20."
        elif n < 1:
            actual_n = 1
            note = f"Requested n={n} is less than minimum allowed (1). Set n to 1."

        slice_n = min(actual_n, total_rows)
        preview_df = df.head(slice_n)

        sample_rows = []
        for idx, row in preview_df.iterrows():
            # Handle text truncation and missing/NaN text
            raw_text = row.get("text", "")
            if pd.isna(raw_text) or raw_text is None:
                text_preview = ""
            else:
                str_text = str(raw_text)
                if len(str_text) > 100:
                    text_preview = str_text[:100] + "..."
                else:
                    text_preview = str_text

            category_val = row.get("category_name", row.get("category", ""))
            if pd.isna(category_val) or category_val is None:
                category_val = ""

            sample_rows.append({
                "index": int(idx),
                "text": text_preview,
                "category_name": str(category_val)
            })

        summary = {
            "result_id": session.next_result_id("inspect_data"),
            "shape": list(df.shape),
            "columns": list(df.columns),
            "n_requested": n,
            "n_shown": len(sample_rows),
            "sample_rows": sample_rows
        }
        if note:
            summary["note"] = note

        session.store_result("inspect_data_tool", {"n": n}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> Dict[str, Any]:
        """List files and directories in a subdirectory of the repository.

        Safely inspects subdirectories within the repository root, rejecting path
        traversal attempts (such as '..' or absolute paths outside the CWD).

        Args:
            subdirectory: Relative path of the directory to list (default root "").

        Returns:
            Dict containing sorted lists of directories and files.
        """
        cwd = Path.cwd().resolve()
        target_path = (cwd / subdirectory).resolve()

        # Security check: ensure target_path is within cwd
        try:
            target_path.relative_to(cwd)
        except ValueError:
            return {
                "error": "Security error: Access outside working directory is forbidden."
            }

        if not target_path.exists():
            return {
                "error": f"Directory '{subdirectory}' does not exist."
            }

        if not target_path.is_dir():
            return {
                "error": f"Path '{subdirectory}' is not a directory."
            }

        directories = []
        files = []

        for entry in target_path.iterdir():
            if entry.name.startswith("."):
                continue  # Skip hidden files/directories
            if entry.is_dir():
                directories.append(entry.name)
            elif entry.is_file():
                files.append(entry.name)

        summary = {
            "result_id": session.next_result_id("list_files"),
            "subdirectory": subdirectory,
            "directories": sorted(directories),
            "files": sorted(files)
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def tokenize_tool(text_column: str = "text") -> Dict[str, Any]:
        """Tokenize text in session.dataframe using NLTK word_tokenize on lowercased text.

        Safely converts missing or NaN values to empty strings, lowercases text,
        and tokenizes using NLTK's word_tokenize. Stores resulting token lists
        in df['unigrams'] and session.artifacts['unigrams'].

        Args:
            text_column: Column name containing document text (default 'text').

        Returns:
            Dict containing document count, total tokens, vocabulary size, and sample tokens.
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

        # Ensure required NLTK data resources are available
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        # Tokenize each text entry safely handling NaNs / non-strings
        def safe_tokenize(val):
            if pd.isna(val) or val is None:
                return []
            text_str = str(val).lower()
            return nltk.word_tokenize(text_str)

        tokens_series = df[text_column].apply(safe_tokenize)
        df["unigrams"] = tokens_series
        session.artifacts["unigrams"] = tokens_series.tolist()

        # Compute summary metrics
        total_documents = len(df)
        all_tokens = [tok for doc in tokens_series for tok in doc]
        total_tokens = len(all_tokens)
        vocab_size = len(set(all_tokens))
        sample_tokens = [doc_tokens[:20] for doc_tokens in tokens_series.head(5).tolist()]

        summary = {
            "result_id": session.next_result_id("tokenize"),
            "n_documents": total_documents,
            "total_tokens": total_tokens,
            "vocab_size": vocab_size,
            "sample_tokens": sample_tokens
        }

        session.store_result("tokenize_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Compute text length statistics and set a category boxplot on session.pending_figure.

        Calculates text length using df['text'].fillna('').astype(str).apply(len),
        computes .describe() statistics overall and per category_name, and creates a
        seaborn boxplot stored on session.pending_figure.

        Returns:
            Dict containing overall and per-category text_length describe statistics.
        """
        if session.dataframe is None:
            return {
                "error": "No dataset loaded. Run load_dataset_tool first."
            }

        df = session.dataframe
        if "text" not in df.columns:
            return {
                "error": "Column 'text' not found in dataframe."
            }

        # Compute text length character counts handling missing values
        df["text_length"] = df["text"].fillna("").astype(str).apply(len)

        # Calculate overall describe statistics
        overall_stats_series = df["text_length"].describe()

        def clean_stats(series_or_dict):
            """Convert stats dict to Python floats and replace NaN with None for JSON compatibility."""
            res = {}
            for k, v in series_or_dict.items():
                if pd.isna(v) or np.isnan(v):
                    res[k] = None
                else:
                    res[k] = float(v) if k != "count" else int(v)
            return res

        overall_stats = clean_stats(overall_stats_series)

        # Calculate per-category describe statistics if category_name column exists
        per_category_stats = {}
        if "category_name" in df.columns:
            grouped = df.groupby("category_name")["text_length"].describe()
            for cat, row in grouped.iterrows():
                per_category_stats[str(cat)] = clean_stats(row)

        # Generate box plot by category
        fig, ax = plt.subplots(figsize=(8, 5))
        if "category_name" in df.columns:
            sns.boxplot(x="category_name", y="text_length", data=df, ax=ax)
            ax.set_title("Document Text Length by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(y="text_length", data=df, ax=ax)
            ax.set_title("Document Text Length Distribution")
        ax.set_ylabel("Text Length (characters)")
        plt.tight_layout()

        # Set pending figure on session
        session.pending_figure = fig

        summary = {
            "result_id": session.next_result_id("describe_data"),
            "overall": overall_stats,
            "per_category": per_category_stats
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    return [
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        inspect_data_tool,
        list_files_tool,
        tokenize_tool,
        describe_data_tool
    ]
