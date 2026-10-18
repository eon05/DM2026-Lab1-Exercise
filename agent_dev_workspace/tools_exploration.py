from typing import Dict, Any
import numpy as np
import pandas as pd
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

    return [cosine_similarity_tool]
