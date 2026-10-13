from common.observability import RequestMetrics


def test_request_metrics_reports_nodes_and_estimated_tokens() -> None:
    metrics = RequestMetrics("req-1")
    metrics.node("intent")
    metrics.llm(40, 20)
    metrics.node("reply")

    result = metrics.finish()

    assert result["request_id"] == "req-1"
    assert result["nodes"] == ["intent", "reply"]
    assert result["llm_calls"] == 1
    assert result["estimated_tokens"] == 15
