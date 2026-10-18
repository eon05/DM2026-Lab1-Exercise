import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import rankdata
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

    @tool
    def pearson_filter_tool(target_class: str, top_k: int = 20) -> dict:
        """Computes Pearson correlation between each term's count and a one-vs-rest binary target class.

        Args:
            target_class: Category label to compute correlation against (one-vs-rest).
            top_k: Number of top-correlated terms to include in the summary sample.

        Returns:
            A dictionary with execution summary, result_id, and top correlated terms.
        """
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {"status": "error", "message": "DTM or labels not found. Run build_dtm_tool first."}

        available_categories = session.categories or list(set(session.labels))
        if target_class not in available_categories:
            return {
                "status": "error",
                "message": f"Target class '{target_class}' not found. Available categories: {available_categories}"
            }

        X = session.feature_matrix
        N = X.shape[0]
        feature_names = np.array(session.feature_names)

        # One-vs-rest target array
        y = (np.array(session.labels) == target_class).astype(float)

        mean_y = y.mean()
        std_y = y.std(ddof=0)

        if std_y == 0:
            pearson_r = np.zeros(len(feature_names))
        else:
            mean_X = np.array(X.mean(axis=0)).ravel()
            mean_sq_X = np.array(X.power(2).mean(axis=0)).ravel()
            std_X = np.sqrt(np.maximum(0.0, mean_sq_X - (mean_X ** 2)))

            # E[X * y]
            E_Xy = np.array(X.T.dot(y)).ravel() / N
            # Cov(X, y) = E[X*y] - E[X]*E[y]
            cov_Xy = E_Xy - (mean_X * mean_y)

            denom = std_X * std_y
            with np.errstate(divide='ignore', invalid='ignore'):
                pearson_r = np.where(denom > 0, cov_Xy / denom, 0.0)

            pearson_r = np.nan_to_num(pearson_r, nan=0.0)

        df_report = pd.DataFrame({
            "term": feature_names,
            "pearson_r": pearson_r
        })

        # Sort by absolute correlation value descending
        df_report["abs_r"] = df_report["pearson_r"].abs()
        df_report = df_report.sort_values(by="abs_r", ascending=False).reset_index(drop=True)
        # Keep term first and pearson_r second as required
        df_report = df_report[["term", "pearson_r"]]

        result_id = session.next_result_id("pearson_filter")
        top_terms = df_report.head(top_k).to_dict(orient="records")

        summary = {
            "result_id": result_id,
            "status": "success",
            "target_class": target_class,
            "total_terms": len(df_report),
            "top_k": top_k,
            "top_correlated_terms": top_terms
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "top_k": top_k},
            summary=summary,
            full_report=df_report
        )

        return summary

    @tool
    def spearman_filter_tool(target_class: str, top_k: int = 20) -> dict:
        """Computes Spearman rank correlation between each term's count and a one-vs-rest binary target class.

        Args:
            target_class: Category label to compute correlation against (one-vs-rest).
            top_k: Number of top-correlated terms to include in the summary sample.

        Returns:
            A dictionary with execution summary, result_id, and top correlated terms.
        """
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {"status": "error", "message": "DTM or labels not found. Run build_dtm_tool first."}

        available_categories = session.categories or list(set(session.labels))
        if target_class not in available_categories:
            return {
                "status": "error",
                "message": f"Target class '{target_class}' not found. Available categories: {available_categories}"
            }

        X = session.feature_matrix
        N, P = X.shape
        feature_names = np.array(session.feature_names)

        y = (np.array(session.labels) == target_class).astype(float)
        rank_y = rankdata(y)

        # Process in memory-safe column chunks for large vocabularies
        spearman_r = np.zeros(P)
        chunk_size = 5000

        for start_col in range(0, P, chunk_size):
            end_col = min(start_col + chunk_size, P)
            X_chunk_dense = X[:, start_col:end_col].toarray()

            for idx in range(X_chunk_dense.shape[1]):
                col_data = X_chunk_dense[:, idx]
                if np.all(col_data == col_data[0]):
                    spearman_r[start_col + idx] = 0.0
                else:
                    rank_x = rankdata(col_data)
                    # Compute Pearson correlation on ranks
                    r = np.corrcoef(rank_x, rank_y)[0, 1]
                    spearman_r[start_col + idx] = 0.0 if np.isnan(r) else r

        df_report = pd.DataFrame({
            "term": feature_names,
            "spearman_r": spearman_r
        })

        # Sort by absolute correlation value descending
        df_report["abs_r"] = df_report["spearman_r"].abs()
        df_report = df_report.sort_values(by="abs_r", ascending=False).reset_index(drop=True)
        # Term first and spearman_r second
        df_report = df_report[["term", "spearman_r"]]

        result_id = session.next_result_id("spearman_filter")
        top_terms = df_report.head(top_k).to_dict(orient="records")

        summary = {
            "result_id": result_id,
            "status": "success",
            "target_class": target_class,
            "total_terms": len(df_report),
            "top_k": top_k,
            "top_correlated_terms": top_terms
        }

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "top_k": top_k},
            summary=summary,
            full_report=df_report
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
