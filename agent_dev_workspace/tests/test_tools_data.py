from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools as make_data_tools
from tools_dtm import make_tools as make_dtm_tools


def _load_fixture(session):
    """Helper to load sample fixture into a session."""
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })


def test_check_missing_tool():
    """Known Answer #2: check_missing_tool on sample_fixture.csv."""
    session = SessionState()
    _load_fixture(session)

    data_tools = make_data_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({"text_column": "text"})

    assert res["n_missing"] == 1
    assert res["missing_indices"] == [7]
    assert res["total_rows"] == 8


def test_check_duplicates_tool_no_drop():
    """Known Answer #3: check_duplicates_tool with drop=False on sample_fixture.csv."""
    session = SessionState()
    _load_fixture(session)

    data_tools = make_data_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    res = check_duplicates_tool.invoke({"drop": False})

    assert res["n_duplicates"] == 1
    assert res["n_rows_dropped"] == 0
    assert res["rows_before"] == 8
    assert res["rows_after"] == 8
    assert len(session.dataframe) == 8


def test_check_duplicates_tool_with_drop_and_dtm_clearing():
    """Known Answer #3 variant: drop=True clears feature_matrix built by build_dtm_tool."""
    session = SessionState()
    _load_fixture(session)

    # Build DTM using actual build_dtm_tool first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Verify DTM was created before dropping duplicates
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape[0] == 8

    data_tools = make_data_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    res = check_duplicates_tool.invoke({"drop": True})

    assert res["n_duplicates"] == 1
    assert res["n_rows_dropped"] == 2
    assert res["rows_before"] == 8
    assert res["rows_after"] == 6
    assert len(session.dataframe) == 6

    # Verify downstream matrix state gets cleared
    assert session.feature_matrix is None
    assert session.feature_names is None
    assert session.labels is None
    assert session.categories == []
    assert res["cleared_downstream_state"] is True


def test_sample_data_tool_and_dtm_clearing():
    """Known Answer #13: sample_data_tool with n=4, random_state=42 on sample_fixture.csv."""
    session = SessionState()
    _load_fixture(session)

    # Build DTM using build_dtm_tool first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Verify DTM was created before sampling
    assert session.feature_matrix is not None

    data_tools = make_data_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert res["n_samples"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert res["total_rows_before"] == 8
    assert res["total_rows_after"] == 4
    assert len(session.dataframe) == 4

    # Verify downstream matrix state gets cleared
    assert session.feature_matrix is None
    assert session.feature_names is None
    assert session.labels is None
    assert session.categories == []
    assert res["cleared_downstream_state"] is True


def test_list_files_tool():
    """Known Answer #11: list_files_tool on newdataset and traversal security check."""
    session = SessionState()

    data_tools = make_data_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # Known answer #11 check
    res = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]

    # Security path traversal check
    res_traversal = list_files_tool.invoke({"subdirectory": "../"})
    assert "error" in res_traversal
    assert "Security error" in res_traversal["error"]
