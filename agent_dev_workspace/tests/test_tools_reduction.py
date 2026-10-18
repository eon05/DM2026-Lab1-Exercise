"""Tests for reduce_dimensions_tool in tools_reduction.py against known answers #7, #8, and #9."""

import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def _setup_session_and_dtm():
    """Helper to initialize fresh SessionState, load sample_fixture.csv, and build DTM."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    dtm_tool = next(t for t in make_dtm_tools(session) if t.name == "build_dtm_tool")

    # Load fixture data
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Build DTM
    dtm_tool.invoke({})

    return session


def test_reduce_dimensions_pca():
    """Known answer #7: PCA reduction to 2D coordinates and explained_variance_ratio."""
    session = _setup_session_and_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({"method": "pca", "n_components": 2})

    # Expected coordinates for known answer #7
    expected_pca_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ])

    expected_explained_var = [0.7532, 0.1965]

    coords = session.artifacts.get("reduced_coordinates")
    assert coords is not None
    assert np.allclose(coords, expected_pca_coords, atol=1e-3)

    # Check explained variance ratio in summary
    assert "explained_variance_ratio" in res
    assert np.allclose(res["explained_variance_ratio"], expected_explained_var, atol=1e-3)

    # Check that pending_figure was created
    assert session.pending_figure is not None


def test_reduce_dimensions_tsne():
    """Known answer #8: t-SNE topological properties test (duplicate equality & cluster separation)."""
    session = _setup_session_and_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({
        "method": "tsne",
        "n_components": 2,
        "perplexity": 2,
        "random_state": 42,
    })

    coords = session.artifacts.get("reduced_coordinates")
    assert coords is not None

    # Check exact match for duplicate rows 5 and 6
    assert np.allclose(coords[5], coords[6], atol=1e-3)

    # Check cluster proximity: catA documents (rows 0..2) should be closer to each other
    # than catA documents are to catB documents (rows 3..6)
    catA_coords = coords[:3]
    catB_coords = coords[3:7]

    # Mean distance within catA
    catA_distances = [
        np.linalg.norm(catA_coords[i] - catA_coords[j])
        for i in range(len(catA_coords))
        for j in range(i + 1, len(catA_coords))
    ]
    mean_intra_catA = np.mean(catA_distances)

    # Mean distance between catA and catB
    inter_cat_distances = [
        np.linalg.norm(a - b)
        for a in catA_coords
        for b in catB_coords
    ]
    mean_inter_cat = np.mean(inter_cat_distances)

    # CatA points should be more tightly clustered together than they are to CatB
    assert mean_intra_catA < mean_inter_cat

    # Check pending_figure
    assert session.pending_figure is not None


def test_reduce_dimensions_umap():
    """Known answer #9: UMAP reduction with n_neighbors=3, random_state=42."""
    session = _setup_session_and_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({
        "method": "umap",
        "n_components": 2,
        "n_neighbors": 3,
        "random_state": 42,
    })

    # Expected coordinates for known answer #9
    expected_umap_coords = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622],
    ])

    coords = session.artifacts.get("reduced_coordinates")
    assert coords is not None
    assert np.allclose(coords, expected_umap_coords, atol=1e-2)

    # Check pending_figure
    assert session.pending_figure is not None
