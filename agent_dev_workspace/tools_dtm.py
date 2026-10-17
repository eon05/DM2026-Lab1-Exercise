from typing import List, Optional, Dict, Any, Union
import numpy as np
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        ngram_range: List[int] = [1, 1],
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
    ) -> Dict[str, Any]:
        """Build a document-term matrix (DTM) from the dataset's text column.

        Extracts unigrams/ngrams into a sparse feature matrix using CountVectorizer
        without removing stopwords. Handles missing or NaN text values by treating them
        as empty strings.

        Args:
            max_features: Optional maximum number of terms to retain by frequency.
            ngram_range: Range of n-gram sizes [min_n, max_n]. Default is [1, 1] (unigrams).
            min_df: Minimum document frequency threshold (integer count or float proportion). Default is 1.
            max_df: Maximum document frequency threshold (integer count or float proportion). Default is 1.0.

        Returns:
            A summary dictionary containing matrix dimensions and sparsity metrics.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe or 'text' column loaded in session."}

        # Ensure missing or NaN text values are filled with empty strings so CountVectorizer
        # processes all rows without crashing or dropping rows.
        texts = session.dataframe["text"].fillna("").astype(str)

        # Do NOT pass stop_words="english" here; build_dtm_tool retains all tokens/stopwords
        # as specified in the pipeline contract and required for known answers.
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=tuple(ngram_range),
            min_df=min_df,
            max_df=max_df,
        )

        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].tolist())

        # Sparsity calculations
        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = (
            float(100.0 * (1.0 - (non_zero / total_elements)))
            if total_elements > 0
            else 0.0
        )

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "shape": [n_rows, n_cols],
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        args = {
            "max_features": max_features,
            "ngram_range": ngram_range,
            "min_df": min_df,
            "max_df": max_df,
        }
        session.store_result("build_dtm_tool", args, summary)
        return summary

    return [build_dtm_tool]
