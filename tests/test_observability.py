import asyncio

from common.observability import ObservedLLM, RequestMetrics, observe_node, observe_request


def test_request_metrics_reports_nodes_and_estimated_tokens() -> None:
    metrics = RequestMetrics("req-1")
    metrics.node("intent")
    metrics.llm(40, 20, 3.5, "intent")
    metrics.response_chars = 8
    metrics.node("reply")

    result = metrics.finish()

    assert result["request_id"] == "req-1"
    assert result["nodes"] == ["intent", "reply"]
    assert result["llm_calls"] == 1
    assert result["estimated_tokens"] == 15
    assert result["response_chars"] == 8
    assert result["llm_spans"][0]["node"] == "intent"


def test_observed_llm_keeps_concurrent_request_metrics_isolated() -> None:
    class DelayedLLM:
        async def complete(self, messages, *, temperature=0.0):
            await asyncio.sleep(0.01)
            return str(messages[0]["content"])

    llm = ObservedLLM(DelayedLLM())
    first = RequestMetrics("first-request")
    second = RequestMetrics("second-request")

    async def call(metrics: RequestMetrics, node: str) -> None:
        with observe_request(metrics), observe_node(node):
            await llm.complete([{"role": "user", "content": node}])

    async def run_calls() -> None:
        await asyncio.gather(call(first, "intent"), call(second, "planner"))

    asyncio.run(run_calls())

    first_result = first.finish()
    second_result = second.finish()
    assert first_result["llm_calls"] == second_result["llm_calls"] == 1
    assert first_result["llm_spans"][0]["node"] == "intent"
    assert second_result["llm_spans"][0]["node"] == "planner"
    assert first_result["input_chars"] == len("intent")
    assert second_result["input_chars"] == len("planner")


def test_observed_llm_records_failures_without_prompt_content() -> None:
    class BrokenLLM:
        async def complete(self, messages, *, temperature=0.0):
            raise RuntimeError("provider unavailable")

    metrics = RequestMetrics("failed-request")
    llm = ObservedLLM(BrokenLLM())

    async def call() -> None:
        with observe_request(metrics), observe_node("recognize_intent"):
            await llm.complete([{"role": "user", "content": "private prompt"}])

    try:
        asyncio.run(call())
    except RuntimeError:
        pass

    result = metrics.finish()
    assert result["llm_errors"] == 1
    assert result["llm_spans"][0]["error_type"] == "RuntimeError"
    assert "private prompt" not in str(result)
