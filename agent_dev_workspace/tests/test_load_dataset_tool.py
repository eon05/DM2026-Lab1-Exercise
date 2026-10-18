"""Tests for premade load_dataset_tool against known answer #1 in TEST_FIXTURE.md."""

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


def test_load_dataset_tool_sample_fixture():
    """Known answer #1: load_dataset_tool on sample_fixture.csv produces 8 documents split 4 and 4."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Assert load_dataset_tool output dict shape matches known answer #1
    assert res["n_documents"] == 8
    assert res["counts_per_category"] == {"catA": 4, "catB": 4}

    # Assert dataframe state
    assert session.dataframe is not None
    assert len(session.dataframe) == 8
