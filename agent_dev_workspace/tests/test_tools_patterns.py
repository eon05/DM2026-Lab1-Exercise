import os
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def _load_fixture_dataset():
    """Helper to load the sample fixture dataset into a fresh SessionState."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    csv_path = os.path.join("agent_dev", "sample_fixture.csv")
    load_dataset_tool.invoke({
        "file_path": csv_path,
        "text_column": "text",
        "label_column": "label",
    })
    
    tools_list = make_tools(session)
    mine_patterns_tool = next(t for t in tools_list if t.name == "mine_patterns_tool")
    return session, mine_patterns_tool


def test_mine_patterns_fpgrowth_tfidf():
    """Tests mine_patterns_tool with algorithm='fpgrowth' and filtering_method='tfidf' (Known Answer #17)."""
    session, mine_patterns_tool = _load_fixture_dataset()

    # catA with min_sup=3 -> finds {alpha} with support 3
    res_catA_3 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 3,
    })
    assert res_catA_3["status"] == "success"
    assert res_catA_3["n_patterns"] == 1
    assert res_catA_3["patterns"][0]["pattern"] == ["alpha"]
    assert res_catA_3["patterns"][0]["support"] == 3
    assert "patterns_catA_tfidf_fpgrowth" in session.artifacts
    assert session.artifacts["patterns_catA_tfidf_fpgrowth"] == res_catA_3["patterns"]

    # catA with min_sup=4 -> finds 0 patterns
    res_catA_4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4,
    })
    assert res_catA_4["status"] == "success"
    assert res_catA_4["n_patterns"] == 0
    assert res_catA_4["patterns"] == []

    # catB with min_sup=4 -> finds {gamma} with support 4
    res_catB_4 = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4,
    })
    assert res_catB_4["status"] == "success"
    assert res_catB_4["n_patterns"] == 1
    assert res_catB_4["patterns"][0]["pattern"] == ["gamma"]
    assert res_catB_4["patterns"][0]["support"] == 4


def test_mine_patterns_zero_surviving_terms():
    """Tests variance and term_frequency filtering where zero terms survive (Known Answer #17)."""
    session, mine_patterns_tool = _load_fixture_dataset()

    for filtering in ["variance", "term_frequency"]:
        for cat in ["catA", "catB"]:
            res = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": filtering,
                "algorithm": "fpgrowth",
                "min_sup": 1,
            })
            assert res["status"] == "success"
            assert res["n_patterns"] == 0
            assert res["patterns"] == []
            assert "no terms survived filtering" in res.get("note", "")


def test_mine_patterns_topk():
    """Tests algorithm='topk' with k=1 (Known Answer #20)."""
    session, mine_patterns_tool = _load_fixture_dataset()

    # catA topk k=1 -> {alpha} support 3
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1,
    })
    assert res_catA["status"] == "success"
    assert res_catA["n_patterns"] == 1
    assert res_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_catA["patterns"][0]["support"] == 3

    # catB topk k=1 -> {gamma} support 4
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1,
    })
    assert res_catB["status"] == "success"
    assert res_catB["n_patterns"] == 1
    assert res_catB["patterns"][0]["pattern"] == ["gamma"]
    assert res_catB["patterns"][0]["support"] == 4


def test_mine_patterns_maxfpgrowth():
    """Tests algorithm='maxfpgrowth' which returns 0 patterns on this dataset (Known Answer #20)."""
    session, mine_patterns_tool = _load_fixture_dataset()

    for cat in ["catA", "catB"]:
        res = mine_patterns_tool.invoke({
            "category_name": cat,
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": 1,
        })
        assert res["status"] == "success"
        assert res["n_patterns"] == 0
        assert res["patterns"] == []


def test_mine_patterns_validation_errors():
    """Tests parameter validation error for algorithm='topk' without k."""
    session, mine_patterns_tool = _load_fixture_dataset()

    res = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
    })
    assert res["status"] == "error"
    assert "k" in res["error"]
