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
