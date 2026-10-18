import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_known_answer_16():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load dataset
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Build DTM using real build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Get cosine_similarity_tool
    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(
        t for t in exploration_tools if t.name == "cosine_similarity_tool"
    )

    # Known answer #16 checks
    # Doc 0 vs 1 = 0.9847
    res_0_1 = cosine_similarity_tool.invoke({"doc1": 0, "doc2": 1})
    assert "error" not in res_0_1
    assert isinstance(res_0_1["similarity"], float)
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-4)

    # Doc 0 vs 3 = 0.0909
    res_0_3 = cosine_similarity_tool.invoke({"doc1": 0, "doc2": 3})
    assert "error" not in res_0_3
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-4)

    # Doc 3 vs 4 = 0.9847
    res_3_4 = cosine_similarity_tool.invoke({"doc1": 3, "doc2": 4})
    assert "error" not in res_3_4
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-4)

    # Doc 0 vs 7 = 0.0 exactly
    res_0_7 = cosine_similarity_tool.invoke({"doc1": 0, "doc2": 7})
    assert "error" not in res_0_7
    assert res_0_7["similarity"] == 0.0


def test_cosine_similarity_missing_dtm_error():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(
        t for t in exploration_tools if t.name == "cosine_similarity_tool"
    )

    res = cosine_similarity_tool.invoke({"doc1": 0, "doc2": 1})
    assert "error" in res
    assert "build_dtm_tool" in res["error"]


def test_cosine_similarity_index_out_of_range():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(
        t for t in exploration_tools if t.name == "cosine_similarity_tool"
    )

    res = cosine_similarity_tool.invoke({"doc1": 0, "doc2": 99})
    assert "error" in res
    assert "out of range" in res["error"].lower()


def test_feature_correlation_matrix_known_answer_19():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(
        t for t in exploration_tools if t.name == "feature_correlation_matrix_tool"
    )

    res = feature_corr_tool.invoke({})
    assert "error" not in res

    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms

    matrix = res["correlation_matrix"]
    # Check diagonal is all 1.0
    for i in range(len(expected_terms)):
        assert matrix[i][i] == pytest.approx(1.0, abs=1e-4)

    # term indices: alpha=0, gamma=1, delta=2, beta=3, always=4
    idx_alpha = expected_terms.index("alpha")
    idx_gamma = expected_terms.index("gamma")
    idx_delta = expected_terms.index("delta")
    idx_beta = expected_terms.index("beta")
    idx_always = expected_terms.index("always")

    # alpha vs beta = 0.8885
    assert matrix[idx_alpha][idx_beta] == pytest.approx(0.8885, abs=1e-4)
    # gamma vs delta = 0.8307
    assert matrix[idx_gamma][idx_delta] == pytest.approx(0.8307, abs=1e-4)
    # alpha vs gamma = -0.5718
    assert matrix[idx_alpha][idx_gamma] == pytest.approx(-0.5718, abs=1e-4)
    # always vs delta = 0.3780
    assert matrix[idx_always][idx_delta] == pytest.approx(0.3780, abs=1e-4)

    # Plot figure should be set
    assert session.pending_figure is not None


def test_feature_correlation_matrix_missing_dtm_error():
    session = SessionState()

    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(
        t for t in exploration_tools if t.name == "feature_correlation_matrix_tool"
    )

    res = feature_corr_tool.invoke({})
    assert "error" in res
    assert "build_dtm_tool" in res["error"]
