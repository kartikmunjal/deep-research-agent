from src.agent_langgraph.graph import route_after_search


def test_route_after_search_triggers_replan_on_majority_failures() -> None:
    route = route_after_search(
        {
            "sub_questions": ["a", "b", "c"],
            "search_failures": ["a", "b"],
            "replan_attempts": 0,
        }
    )

    assert route == "replan_node"


def test_route_after_search_skips_replan_after_one_attempt() -> None:
    route = route_after_search(
        {
            "sub_questions": ["a", "b", "c"],
            "search_failures": ["a", "b", "c"],
            "replan_attempts": 1,
        }
    )

    assert route == "synthesize_node"
