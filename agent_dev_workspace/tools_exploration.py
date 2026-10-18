from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity


def make_tools(session):
    @tool
    def cosine_similarity_tool(doc1: int, doc2: int) -> Dict[str, Any]:
        """Compute cosine similarity between two documents in the dataset.

        Uses the saved CountVectorizer from build_dtm_tool to transform the text
        of the specified document indices and computes cosine similarity.

        Args:
            doc1: Row index of the first document (0-indexed).
            doc2: Row index of the second document (0-indexed).

        Returns:
            A summary dictionary containing doc1, doc2, previews of their text,
            and the similarity score rounded to 4 decimal places.
        """
        vectorizer = session.artifacts.get("count_vectorizer")
        if session.feature_matrix is None or vectorizer is None:
            return {
                "error": "Document-term matrix or vectorizer not found. Please run build_dtm_tool first."
            }

        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe or 'text' column loaded in session."}

        n_docs = len(session.dataframe)
        if not (0 <= doc1 < n_docs) or not (0 <= doc2 < n_docs):
            return {
                "error": f"Document index out of range. Valid range is 0 to {n_docs - 1}."
            }

        texts = session.dataframe["text"].fillna("").astype(str)
        text1 = texts.iloc[doc1]
        text2 = texts.iloc[doc2]

        v1 = vectorizer.transform([text1])
        v2 = vectorizer.transform([text2])

        sim_matrix = cosine_similarity(v1, v2)
        raw_score = sim_matrix[0, 0]

        if np.isnan(raw_score):
            similarity_score = 0.0
        else:
            similarity_score = round(float(raw_score), 4)

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc1": doc1,
            "doc2": doc2,
            "text1_preview": text1[:50],
            "text2_preview": text2[:50],
            "similarity": similarity_score,
        }

        args = {"doc1": doc1, "doc2": doc2}
        session.store_result("cosine_similarity_tool", args, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool() -> Dict[str, Any]:
        """Compute Pearson correlation matrix across top terms by variance.

        Selects up to 20 terms with the highest variance from the global document-term
        matrix, computes their Pearson correlation matrix, sets an annotated seaborn
        heatmap on session.pending_figure, and returns the terms and matrix values.

        Returns:
            A summary dictionary containing result_id, terms (in variance-descending order),
            and the correlation_matrix as a nested list of floats/None.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "Document-term matrix not found. Please run build_dtm_tool first."
            }

        dense_matrix = session.feature_matrix.toarray()
        feature_names = np.array(session.feature_names)

        # Variance per term
        variances = dense_matrix.var(axis=0)

        # Top 20 terms by variance descending
        sorted_indices = np.argsort(-variances)
        top_k = min(20, len(feature_names))
        top_indices = sorted_indices[:top_k]

        top_terms = feature_names[top_indices].tolist()
        top_matrix = dense_matrix[:, top_indices]

        # Pearson correlation matrix across top terms
        corr_matrix = np.corrcoef(top_matrix, rowvar=False)

        # Format correlation matrix into nested list with None for NaNs and rounded to 4 decimals
        formatted_matrix: List[List[Optional[float]]] = []
        if top_k == 1:
            # np.corrcoef on 1D/single feature produces a scalar or 2x2 depending on call,
            # but for 1 feature corr is 1.0 (or NaN if 0 variance)
            val = float(corr_matrix) if not np.isnan(corr_matrix) else None
            formatted_matrix = [[1.0 if val is not None else None]]
            corr_matrix = np.array([[1.0 if val is not None else np.nan]])
        else:
            for row in corr_matrix:
                formatted_row: List[Optional[float]] = []
                for val in row:
                    if np.isnan(val):
                        formatted_row.append(None)
                    else:
                        formatted_row.append(round(float(val), 4))
                formatted_matrix.append(formatted_row)

        # Generate annotated seaborn heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax,
        )
        ax.set_title("Top Term Correlation Matrix")
        plt.tight_layout()
        session.pending_figure = fig

        # Prepare full_report as DataFrame
        full_report_df = pd.DataFrame(corr_matrix, index=top_terms, columns=top_terms)

        result_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": result_id,
            "terms": top_terms,
            "correlation_matrix": formatted_matrix,
        }

        session.store_result(
            "feature_correlation_matrix_tool",
            {},
            summary,
            full_report=full_report_df,
        )
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
