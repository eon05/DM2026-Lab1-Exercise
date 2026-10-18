"""Tools for dimensionality reduction (PCA, t-SNE, UMAP) and label binarization."""

from typing import Any, Dict, List, Literal, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def make_tools(session: Any) -> List[Any]:
    """Factory function that returns dimensionality reduction tools bound to session state."""

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Reduce feature matrix dimensions using PCA, t-SNE, or UMAP to 2D coordinates and generate a scatter plot.

        Args:
            method: Reduction algorithm to use ("pca", "tsne", or "umap"). Default "pca".
            n_components: Target number of dimensions (must be 2). Default 2.
            perplexity: t-SNE perplexity parameter. Default 30.0.
            n_neighbors: UMAP n_neighbors parameter. Default 15.
            random_state: Random seed for reproducibility. Default 42.

        Returns:
            Dict containing summary of reduced coordinates, method parameters, and optional explained variance ratio.
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "No feature matrix found in session. Run build_dtm_tool first."}

        # Convert sparse feature matrix to dense numpy array (matches Master implementation)
        if hasattr(session.feature_matrix, "toarray"):
            X = session.feature_matrix.toarray()
        else:
            X = np.asarray(session.feature_matrix)

        n_samples = X.shape[0]
        if n_samples == 0:
            return {"status": "error", "message": "Feature matrix has 0 samples."}

        method_clean = method.lower().strip()
        perplexity_adjusted = None
        neighbors_adjusted = None

        if method_clean == "pca":
            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(X)
            explained_variance_ratio = model.explained_variance_ratio_.tolist()
        elif method_clean == "tsne":
            # Perplexity must be strictly less than n_samples in scikit-learn
            actual_perplexity = perplexity
            if perplexity >= n_samples:
                actual_perplexity = max(1.0, float(n_samples - 1))
                perplexity_adjusted = (
                    f"Perplexity ({perplexity}) was >= n_samples ({n_samples}); "
                    f"adjusted to {actual_perplexity}."
                )

            model = TSNE(
                n_components=n_components,
                perplexity=actual_perplexity,
                random_state=random_state,
            )
            coords = model.fit_transform(X)
            explained_variance_ratio = None
        elif method_clean == "umap":
            try:
                import umap
            except ImportError:
                return {
                    "status": "error",
                    "message": "umap-learn package is not installed. Install it with `pip install umap-learn`.",
                }

            # Adjust n_neighbors if larger than n_samples
            actual_n_neighbors = n_neighbors
            if n_neighbors >= n_samples:
                actual_n_neighbors = max(2, n_samples - 1)
                neighbors_adjusted = (
                    f"n_neighbors ({n_neighbors}) was >= n_samples ({n_samples}); "
                    f"adjusted to {actual_n_neighbors}."
                )

            model = umap.UMAP(
                n_components=n_components,
                n_neighbors=actual_n_neighbors,
                random_state=random_state,
            )
            coords = model.fit_transform(X)
            explained_variance_ratio = None
        else:
            return {
                "status": "error",
                "message": f"Unknown method '{method}'. Choose from 'pca', 'tsne', or 'umap'.",
            }

        # Save full coordinates to session artifacts
        session.artifacts["reduced_coordinates"] = coords

        # Generate scatter plot colored by category
        fig, ax = plt.subplots(figsize=(8, 6))

        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            categories = session.dataframe["category_name"].values
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            categories = session.dataframe["category"].values
        else:
            categories = np.zeros(n_samples, dtype=int)

        unique_cats = np.unique(categories)
        for cat in unique_cats:
            mask = categories == cat
            ax.scatter(coords[mask, 0], coords[mask, 1], label=str(cat), alpha=0.7, edgecolors="none")

        ax.set_title(f"{method_clean.upper()} 2D Dimensionality Reduction")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(title="Category")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Store figure in session.pending_figure (no plt.show())
        session.pending_figure = fig

        # Store result in session results store
        res_id = session.next_result_id("reduction")

        summary: Dict[str, Any] = {
            "result_id": res_id,
            "method": method_clean,
            "n_components": n_components,
            "n_samples": n_samples,
            "coordinates_preview": coords[:5].tolist(),
        }

        if method_clean == "pca" and explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        if perplexity_adjusted:
            summary["perplexity_note"] = perplexity_adjusted

        if neighbors_adjusted:
            summary["neighbors_note"] = neighbors_adjusted

        full_report = pd.DataFrame(coords, columns=[f"Component_{i+1}" for i in range(n_components)])
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            full_report["category_name"] = session.dataframe["category_name"].values

        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={
                "method": method,
                "n_components": n_components,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
                "random_state": random_state,
            },
            summary=summary,
            full_report=full_report,
        )

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encodes the category_name column in session.dataframe into binary 0/1 vectors.

        Returns:
            Dict containing result_id, sorted classes, class counts, matrix shape, and preview of vectors.
        """
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"error": "No dataset loaded or 'category_name' column missing."}

        # One-hot encode sorted category names using pd.get_dummies to guarantee one column per class
        classes = sorted([str(c) for c in session.dataframe["category_name"].unique()])
        one_hot_df = pd.get_dummies(session.dataframe["category_name"])[classes].astype(int)

        # Create binary row vectors as list of plain Python ints
        bin_vectors = one_hot_df.values.tolist()

        # Add bin_category column to session.dataframe
        session.dataframe["bin_category"] = bin_vectors

        # Compute counts per class as plain Python ints
        counts_per_class = {cls: int(one_hot_df[cls].sum()) for cls in classes}

        # Build preview of first 5 rows
        preview = []
        for i in range(min(5, len(session.dataframe))):
            preview.append(
                {
                    "index": i,
                    "category_name": str(session.dataframe["category_name"].iloc[i]),
                    "vector": [int(x) for x in bin_vectors[i]],
                }
            )

        res_id = session.next_result_id("binarize")

        summary: Dict[str, Any] = {
            "result_id": res_id,
            "classes": classes,
            "n_classes": len(classes),
            "shape": [len(session.dataframe), len(classes)],
            "counts_per_class": counts_per_class,
            "preview": preview,
        }

        session.store_result(
            tool_name="binarize_labels_tool",
            args={},
            summary=summary,
            full_report=one_hot_df,
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
