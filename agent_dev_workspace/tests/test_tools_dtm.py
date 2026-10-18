import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    # 1. Initialize session and load premade load_dataset_tool
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample fixture CSV
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Instantiate build_dtm_tool from tools_dtm using the SAME session
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Invoke build_dtm_tool
    result = build_dtm_tool.invoke({})

    # 3. Assertions against known answer #4 in TEST_FIXTURE.md
    assert result["shape"] == [8, 5]
    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert pytest.approx(result["sparsity_pct"], abs=1e-3) == 47.5

    # Confirm vocabulary explicitly includes 'always' (stopwords not removed) and matches exact order
    expected_vocabulary = ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_names == expected_vocabulary

    # Confirm artifacts store fitted count_vectorizer
    assert "count_vectorizer" in session.artifacts
    assert session.artifacts["count_vectorizer"] is not None

    # Confirm feature_matrix shape is 8x5
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)

    # Confirm row 7 (8th document, the empty text row) is all zeros
    empty_row_counts = session.feature_matrix.getrow(7).toarray().flatten()
    assert (empty_row_counts == 0).all()


def test_term_frequency_tool_known_answer_14():
    # 1. Initialize a fresh session and load premade load_dataset_tool
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Instantiate tools_dtm tools
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # Error handling check: invoke term_frequency_tool before build_dtm_tool
    error_res = term_frequency_tool.invoke({})
    assert "error" in error_res

    # Build DTM
    build_dtm_tool.invoke({})

    # 3. Invoke term_frequency_tool
    res = term_frequency_tool.invoke({"top_n": 20})

    # Assertions against known answer #14 in TEST_FIXTURE.md
    top_terms = res["top_terms"]
    expected_counts = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }

    assert top_terms == expected_counts

    # Check tie-breaker order: always (count 7) vs gamma (count 7)
    # Convert dictionary keys to list to verify ordering
    term_order = list(top_terms.keys())
    assert term_order.index("always") < term_order.index("gamma")

    # Check values are standard python ints
    for val in top_terms.values():
        assert type(val) is int


def test_dtm_heatmap_tool_known_answer_21():
    # 1. Initialize a fresh session and load dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Instantiate tools
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # Build DTM
    build_dtm_tool.invoke({})

    # 3. Default call (returns full 8x5 matrix)
    res_default = dtm_heatmap_tool.invoke({})

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    expected_vocabulary = ["alpha", "always", "beta", "delta", "gamma"]

    assert res_default["matrix"] == expected_full_matrix
    assert res_default["term_labels"] == expected_vocabulary
    assert res_default["document_labels"] == ["0", "1", "2", "3", "4", "5", "6", "7"]

    # Assert pending figure was set
    assert session.pending_figure is not None

    # 4. Slice call (n_terms=2, n_documents=3)
    res_slice = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert res_slice["matrix"] == [[3, 1], [2, 1], [1, 1]]
    assert res_slice["term_labels"] == ["alpha", "always"]
    assert res_slice["document_labels"] == ["0", "1", "2"]
