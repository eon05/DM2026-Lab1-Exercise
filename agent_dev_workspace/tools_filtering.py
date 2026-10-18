import numpy as np
import pandas as pd
from scipy.sparse import issparse
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0, remove_low_variance: bool = False) -> dict:
        """Computes variance for each term across all documents in the DTM and flags low-variance terms.

        Args:
            threshold: Terms with variance strictly less than this threshold are flagged.
            remove_low_variance: If True, mutates session.feature_matrix and session.feature_names
                to retain only terms with variance >= threshold. Defaults to False.

        Returns:
            A dictionary with execution summary, result_id, and low-variance term details.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No feature matrix found. Run build_dtm_tool first."}

        X = session.feature_matrix
        feature_names = np.array(session.feature_names)

        # Compute mean and variance on sparse matrix (ddof=0)
        # E[X]
        mean = np.array(X.mean(axis=0)).ravel()
        # E[X^2]
        mean_sq = np.array(X.power(2).mean(axis=0)).ravel()
        # Var[X] = E[X^2] - (E[X])^2, clipped at 0 to avoid tiny negative floating point artifacts
        variances = np.maximum(0.0, mean_sq - (mean ** 2))

        # Build full_report DataFrame: term first, variance second
        df_report = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        })
        # Sort descending by variance so highest variance/most informative terms are plotted first
        df_report = df_report.sort_values(by="variance", ascending=False).reset_index(drop=True)

        low_var_mask = df_report["variance"] < threshold
        low_var_df = df_report[low_var_mask]
        all_low_var_terms = low_var_df["term"].tolist()
        n_removed = len(all_low_var_terms)
        n_kept = len(df_report) - n_removed

        if remove_low_variance and n_removed > 0:
            keep_mask = variances >= threshold
            session.feature_matrix = X[:, keep_mask]
            session.feature_names = feature_names[keep_mask].tolist()

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "status": "success",
            "threshold": threshold,
            "total_terms": len(df_report),
            "n_kept": n_kept,
            "n_low_variance": n_removed,
            "low_variance_sample": all_low_var_terms[:20],
            "removed_from_session": remove_low_variance and n_removed > 0
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold, "remove_low_variance": remove_low_variance},
            summary=summary,
            full_report=df_report
        )

        return summary

    return [variance_filter_tool]
