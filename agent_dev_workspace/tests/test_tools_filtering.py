import pytest
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_default():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load workspace tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # 1. Load sample dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Build DTM
    build_dtm_tool.invoke({})

    # 3. Call variance_filter_tool with threshold=0.15, default remove_low_variance=False
    res = variance_filter_tool.invoke({"threshold": 0.15})

    assert res["status"] == "success"
    assert res["n_low_variance"] == 1
    assert res["n_kept"] == 4
    assert res["low_variance_sample"] == ["always"]
    assert res["removed_from_session"] is False

    # Check that session.feature_matrix still has all 5 columns
    assert session.feature_matrix.shape[1] == 5

    # Check full_report stored in results_store
    result_id = res["result_id"]
    stored = session.results_store[result_id]
    full_report = stored["full_report"]

    # Verify column layout: first column is term, second column is variance
    assert list(full_report.columns[:2]) == ["term", "variance"]

    # Check per-term variances against known answer #5
    var_dict = dict(zip(full_report["term"], full_report["variance"]))
    assert var_dict["alpha"] == pytest.approx(1.1875, abs=1e-4)
    assert var_dict["always"] == pytest.approx(0.109375, abs=1e-4)
    assert var_dict["beta"] == pytest.approx(0.234375, abs=1e-4)
    assert var_dict["delta"] == pytest.approx(0.25, abs=1e-4)
    assert var_dict["gamma"] == pytest.approx(1.109375, abs=1e-4)


def test_variance_filter_remove():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})

    res = variance_filter_tool.invoke({"threshold": 0.15, "remove_low_variance": True})

    assert res["status"] == "success"
    assert res["removed_from_session"] is True

    # Check that session feature matrix and names mutated
    assert session.feature_matrix.shape[1] == 4
    assert "always" not in session.feature_names
    assert set(session.feature_names) == {"alpha", "beta", "delta", "gamma"}


def test_pearson_filter_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})

    # Check error handling for invalid target class
    err_res = pearson_filter_tool.invoke({"target_class": "catC"})
    assert err_res["status"] == "error"
    assert "catC" in err_res["message"]

    # Test valid target_class="catB" against known answer #6
    res = pearson_filter_tool.invoke({"target_class": "catB"})
    assert res["status"] == "success"

    result_id = res["result_id"]
    full_report = session.results_store[result_id]["full_report"]

    # Verify column layout: first column is term, second is pearson_r
    assert list(full_report.columns[:2]) == ["term", "pearson_r"]

    r_dict = dict(zip(full_report["term"], full_report["pearson_r"]))
    assert r_dict["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert r_dict["always"] == pytest.approx(0.3780, abs=1e-3)
    assert r_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert r_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert r_dict["gamma"] == pytest.approx(0.8307, abs=1e-3)


def test_spearman_filter_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})

    # Check error handling for invalid target class
    err_res = spearman_filter_tool.invoke({"target_class": "catC"})
    assert err_res["status"] == "error"
    assert "catC" in err_res["message"]

    # Test valid target_class="catB" against known answer #6
    res = spearman_filter_tool.invoke({"target_class": "catB"})
    assert res["status"] == "success"

    result_id = res["result_id"]
    full_report = session.results_store[result_id]["full_report"]

    # Verify column layout: first column is term, second is spearman_r
    assert list(full_report.columns[:2]) == ["term", "spearman_r"]

    r_dict = dict(zip(full_report["term"], full_report["spearman_r"]))
    assert r_dict["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert r_dict["always"] == pytest.approx(0.3780, abs=1e-3)
    assert r_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert r_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert r_dict["gamma"] == pytest.approx(0.9363, abs=1e-3)
