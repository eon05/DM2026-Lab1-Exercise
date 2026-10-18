from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    @tool
    def term_frequency_tool(top_n: int = 20) -> Dict[str, Any]:
        """Aggregate term frequencies across all documents in the document-term matrix.

        Sums column counts in the DTM across all documents, sorts terms by frequency
        descending (breaking ties alphabetically), and returns top N term counts along
        with storing a full pandas DataFrame report for visualization.

        Args:
            top_n: Number of top frequent terms to return in the summary. Default is 20.

        Returns:
            A summary dictionary containing result_id, top terms dict, and total terms count.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "Document-term matrix has not been built yet. Please run build_dtm_tool first."
            }

        # Sum columns across all documents (axis 0)
        col_sums = np.asarray(session.feature_matrix.sum(axis=0)).ravel()

        # Pair feature names with counts and cast counts to standard python int
        term_counts = [
            (term, int(count))
            for term, count in zip(session.feature_names, col_sums)
        ]

        # Sort primarily by frequency descending, secondarily by term alphabetically
        term_counts.sort(key=lambda x: (-x[1], x[0]))

        # Top N terms dict
        top_terms_dict = {term: count for term, count in term_counts[:top_n]}

        # Full report DataFrame (term first, frequency second) for visualize_result_tool
        full_report = pd.DataFrame(term_counts, columns=["term", "frequency"])

        result_id = session.next_result_id("term_freq")
        summary = {
            "result_id": result_id,
            "top_terms": top_terms_dict,
            "total_terms": len(term_counts),
        }

        args = {"top_n": top_n}
        session.store_result(
            "term_frequency_tool", args, summary, full_report=full_report
        )
        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20, n_documents: int = 20
    ) -> Dict[str, Any]:
        """Generate a heatmap of a slice of the raw Document-Term Matrix (DTM).

        Slices the first n_documents rows and n_terms columns of the DTM in their existing order
        without sorting or filtering. Creates an annotated heatmap using seaborn and sets it
        on session.pending_figure.

        Args:
            n_terms: Number of terms (columns) to include from the start of feature_names. Default is 20.
            n_documents: Number of documents (rows) to include from the start of the matrix. Default is 20.

        Returns:
            A summary dictionary containing result_id, the nested list matrix, term_labels, and document_labels.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "Document-term matrix has not been built yet. Please run build_dtm_tool first."
            }

        total_docs, total_terms = session.feature_matrix.shape
        actual_docs = min(n_documents, total_docs)
        actual_terms = min(n_terms, total_terms)

        sub_matrix = session.feature_matrix[:actual_docs, :actual_terms].toarray()
        matrix_list = [[int(val) for val in row] for row in sub_matrix]

        term_labels = session.feature_names[:actual_terms]
        doc_labels = [str(i) for i in range(actual_docs)]

        fig, ax = plt.subplots(figsize=(max(6, actual_terms * 0.8), max(4, actual_docs * 0.5)))
        sns.heatmap(
            sub_matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=term_labels,
            yticklabels=doc_labels,
            ax=ax,
        )
        ax.set_title(f"DTM Heatmap ({actual_docs} docs x {actual_terms} terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "matrix": matrix_list,
            "term_labels": term_labels,
            "document_labels": doc_labels,
        }

        args = {"n_terms": n_terms, "n_documents": n_documents}
        session.store_result("dtm_heatmap_tool", args, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
