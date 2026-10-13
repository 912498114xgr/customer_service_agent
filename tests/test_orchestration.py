import asyncio
import time

import pytest

from customer_care.orchestration import MultiSkillExecutor, SkillTask, TaskPlan
from customer_care.orchestration import MultiSkillPlanner
from common.mock_llm import MockLLM
from customer_care.tools.registry import ToolCatalog, ToolRegistry
from customer_care.graph import build_customer_care_graph
from customer_care.orchestration.result_validation import ToolResultError, validate_tool_result
from customer_care.state import CareState


class RecordingRegistry(ToolRegistry):
    def __init__(self, *, fail_tools: set[str] | None = None, delay: float = 0.0, catalog: ToolCatalog | None = None) -> None:
        super().__init__(catalog)
        self.calls: list[str] = []
        self.fail_tools = fail_tools or set()
        self.delay = delay

    def execute(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        if self.delay:
            time.sleep(self.delay)
        self.calls.append(name)
        if name in self.fail_tools:
            raise RuntimeError(f"injected failure: {name}")
        if name == "get_order_status":
            return {"order_id": arguments.get("order_id"), "status": "delivered"}
        if name == "check_refund_eligibility":
            return {"order_id": arguments.get("order_id"), "eligible_for_review": True, "money_moved": False}
        return {"tool": name, "status": "demo_result", "arguments": arguments, "side_effects": False}


class WrongTargetRegistry(RecordingRegistry):
    def execute(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        if name == "get_order_status":
            self.calls.append(name)
            return {"order_id": "OTHER", "status": "delivered"}
        return super().execute(name, arguments)


class MalformedRegistry(RecordingRegistry):
    def execute(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        self.calls.append(name)
        return {"order_id": arguments.get("order_id")}


def test_independent_read_tasks_run_in_parallel() -> None:
    registry = RecordingRegistry(delay=0.1)
    plan = TaskPlan(tasks=[
        SkillTask(task_id="order", intent="get_order_status", tool_name="get_order_status", capability="get_order_status", arguments={"order_id": "A"}),
        SkillTask(task_id="coupon", intent="get_coupon_status", tool_name="get_coupon_status", capability="get_coupon_status", arguments={"coupon_id": "C"}),
    ])

    started = time.perf_counter()
    report = asyncio.run(MultiSkillExecutor(registry).execute(plan, request_id="r1"))
    elapsed = time.perf_counter() - started

    assert report.status == "succeeded"
    assert set(registry.calls) == {"get_order_status", "get_coupon_status"}
    assert elapsed < 0.19


def test_side_effect_task_requires_confirmation_before_execution() -> None:
    registry = RecordingRegistry()
    plan = TaskPlan(tasks=[SkillTask(task_id="refund", intent="create_refund_review_ticket", tool_name="create_refund_review_ticket", capability="create_refund_review_ticket", arguments={"order_id": "R-1", "reason": "late"}, requires_confirmation=True)])

    report = asyncio.run(MultiSkillExecutor(registry).execute(plan, request_id="r2"))

    assert report.status == "awaiting_confirmation"
    assert registry.calls == []


def test_failed_dependency_skips_downstream_task() -> None:
    registry = RecordingRegistry(fail_tools={"get_order_status"})
    plan = TaskPlan(tasks=[
        SkillTask(task_id="status", intent="get_order_status", tool_name="get_order_status", capability="get_order_status", arguments={"order_id": "A"}),
        SkillTask(task_id="refund", intent="check_refund_eligibility", tool_name="check_refund_eligibility", capability="check_refund_eligibility", arguments={"order_id": "A"}, dependencies=["status"]),
    ])

    report = asyncio.run(MultiSkillExecutor(registry).execute(plan, request_id="r3"))

    assert report.status == "partial"
    assert {item.task_id: item.status for item in report.results} == {"status": "failed", "refund": "skipped"}


def test_tool_timeout_is_reported_and_downstream_is_not_executed() -> None:
    registry = RecordingRegistry(delay=0.2)
    plan = TaskPlan(tasks=[
        SkillTask(task_id="status", intent="get_order_status", tool_name="get_order_status", capability="get_order_status", arguments={"order_id": "A"}, timeout_seconds=0.05),
        SkillTask(task_id="refund", intent="check_refund_eligibility", tool_name="check_refund_eligibility", capability="check_refund_eligibility", arguments={"order_id": "A"}, dependencies=["status"]),
    ])

    report = asyncio.run(MultiSkillExecutor(registry).execute(plan, request_id="timeout"))

    assert report.results[0].status == "failed"
    assert "Timeout" in (report.results[0].error or "") or "timeout" in (report.results[0].error or "").lower()
    assert report.results[1].status == "skipped"


def test_same_request_is_idempotent() -> None:
    registry = RecordingRegistry()
    executor = MultiSkillExecutor(registry)
    plan = TaskPlan(tasks=[SkillTask(task_id="status", intent="get_order_status", tool_name="get_order_status", capability="get_order_status", arguments={"order_id": "A"})])

    first = asyncio.run(executor.execute(plan, request_id="r4"))
    second = asyncio.run(executor.execute(plan, request_id="r4"))

    assert first.status == "succeeded"
    assert second.status == "cached"
    assert registry.calls == ["get_order_status"]


def test_cached_result_does_not_bypass_current_task_contract() -> None:
    registry = RecordingRegistry()
    executor = MultiSkillExecutor(registry)
    valid = TaskPlan(tasks=[SkillTask(
        task_id="lookup",
        intent="get_order_status",
        tool_name="get_order_status",
        capability="get_order_status",
        arguments={"order_id": "A"},
    )])
    forged = TaskPlan(tasks=[SkillTask(
        task_id="lookup",
        intent="get_order_receipt",
        tool_name="get_order_status",
        capability="get_order_status",
        arguments={"order_id": "A"},
    )])

    first = asyncio.run(executor.execute(valid, request_id="cache-contract"))
    second = asyncio.run(executor.execute(forged, request_id="cache-contract"))

    assert first.status == "succeeded"
    assert second.status == "failed"
    assert second.results[0].failure_code == "invalid_plan"
    assert registry.calls == ["get_order_status"]


def test_cycle_is_rejected() -> None:
    with pytest.raises(ValueError, match="循环依赖"):
        TaskPlan(tasks=[
            SkillTask(task_id="a", intent="get_order_status", tool_name="get_order_status", capability="get_order_status", dependencies=["b"]),
            SkillTask(task_id="b", intent="get_order_details", tool_name="get_order_details", capability="get_order_details", dependencies=["a"]),
        ])


def test_planner_marks_high_risk_side_effect() -> None:
    llm = MockLLM([
        '{"tasks":[{"task_id":"status","intent":"create_refund_review_ticket","tool_name":"create_refund_review_ticket","capability":"create_refund_review_ticket","arguments":{"order_id":"R-1","reason":"late"},"dependencies":[]}],"rationale":"需要人工审核","confidence":0.95}'
    ])
    state = CareState(user_id="u1", intent="refund", entities={"order_id": "R-1"}, messages=[{"role": "user", "content": "请审核退款"}])

    decision = asyncio.run(MultiSkillPlanner(llm).plan(state))

    assert decision.plan is not None
    assert decision.plan.requires_confirmation is True
    assert decision.plan.tasks[0].risk_level == "high"


def test_planner_rejects_task_intent_that_selected_skill_cannot_satisfy() -> None:
    llm = MockLLM([
        '{"tasks":[{"task_id":"receipt","intent":"get_order_receipt","tool_name":"get_order_status","capability":"get_order_status","arguments":{"order_id":"A"},"dependencies":[]}],"confidence":0.99}'
    ])
    state = CareState(user_id="u1", intent="order", messages=[{"role": "user", "content": "Show the receipt for order A"}])

    decision = asyncio.run(MultiSkillPlanner(llm).plan(state))

    assert decision.needs_clarification is True
    assert decision.plan is None


def test_wrong_capability_is_rejected_before_tool_execution() -> None:
    registry = RecordingRegistry()
    plan = TaskPlan(tasks=[SkillTask(
        task_id="receipt",
        intent="get_order_receipt",
        tool_name="get_order_status",
        capability="get_order_receipt",
        arguments={"order_id": "A"},
    )])

    report = asyncio.run(MultiSkillExecutor(registry).execute(plan, request_id="wrong-capability"))

    assert report.status == "failed"
    assert report.results[0].failure_code == "invalid_plan"
    assert registry.calls == []


def test_invalid_result_schema_fails_without_retrying_same_tool() -> None:
    registry = MalformedRegistry()
    plan = TaskPlan(tasks=[SkillTask(
        task_id="status",
        intent="get_order_status",
        tool_name="get_order_status",
        capability="get_order_status",
        arguments={"order_id": "A"},
        max_attempts=3,
    )])

    report = asyncio.run(MultiSkillExecutor(registry).execute(plan, request_id="bad-schema"))

    assert report.status == "failed"
    assert report.results[0].failure_code == "invalid_result"
    assert report.results[0].attempts == 1
    assert registry.calls == ["get_order_status"]


@pytest.mark.parametrize(
    ("tool_name", "intent", "arguments", "output", "message"),
    [
        ("get_refund_status", "get_refund_status", {"order_id": "R-1"}, {"order_id": "R-1", "status": "unknown", "estimated_days": 3}, "未知 status"),
        ("check_refund_eligibility", "check_refund_eligibility", {"order_id": "R-1"}, {"order_id": "R-1", "eligible_for_review": True, "money_moved": True}, "安全约束"),
        ("get_refund_status", "get_refund_status", {"order_id": "R-1"}, {"order_id": "R-1", "status": "processing", "estimated_days": -2}, "范围约束"),
    ],
)
def test_result_contract_rejects_unknown_enum_and_unsafe_literal(
    tool_name: str,
    intent: str,
    arguments: dict[str, str],
    output: dict[str, object],
    message: str,
) -> None:
    task = SkillTask(task_id="check", intent=intent, tool_name=tool_name, capability=tool_name, arguments=arguments)

    with pytest.raises(ToolResultError, match=message):
        validate_tool_result(task, output, ToolCatalog.load_default())


def test_wrong_target_result_replans_once_and_preserves_completed_sibling() -> None:
    catalog = ToolCatalog([
        tool.model_copy(update={"capability": "order_lookup", "task_intents": ["order_lookup"]})
        if tool.name in {"get_order_status", "get_order_details"}
        else tool
        for tool in ToolCatalog.load_default().tools
    ])
    registry = WrongTargetRegistry(catalog=catalog)
    plan = TaskPlan(tasks=[
        SkillTask(task_id="status", intent="order_lookup", tool_name="get_order_status", capability="order_lookup", arguments={"order_id": "A"}),
        SkillTask(task_id="details", intent="order_lookup", tool_name="get_order_details", capability="order_lookup", arguments={"order_id": "A"}),
    ])
    replanned = TaskPlan(tasks=[SkillTask(
        task_id="status",
        intent="order_lookup",
        tool_name="get_order_details",
        capability="order_lookup",
        arguments={"order_id": "A"},
    )])
    replans: list[str] = []

    async def replan(task: SkillTask, error: str) -> TaskPlan:
        replans.append(error)
        return replanned

    report = asyncio.run(MultiSkillExecutor(registry, catalog).execute(
        plan,
        request_id="target-mismatch",
        max_replans=1,
        replan=replan,
    ))

    assert report.status == "succeeded"
    assert report.replan_count == 1
    assert report.executed_task_ids == ["status", "details"]
    assert registry.calls.count("get_order_details") == 2
    assert len(replans) == 1 and "target_mismatch" in replans[0]


def test_replan_cannot_change_the_original_task_goal() -> None:
    registry = WrongTargetRegistry()
    plan = TaskPlan(tasks=[SkillTask(
        task_id="status",
        intent="get_order_status",
        tool_name="get_order_status",
        capability="get_order_status",
        arguments={"order_id": "A"},
    )])
    unrelated = TaskPlan(tasks=[SkillTask(
        task_id="status",
        intent="get_order_receipt",
        tool_name="get_order_receipt",
        capability="get_order_receipt",
        arguments={"order_id": "A"},
    )])

    report = asyncio.run(MultiSkillExecutor(registry).execute(
        plan,
        request_id="wrong-replan-goal",
        max_replans=1,
        replan=lambda task, error: asyncio.sleep(0, result=unrelated),
    ))

    assert report.status == "failed"
    assert report.replan_count == 1
    assert any("同一任务目标" in error for error in report.errors)
    assert registry.calls == ["get_order_status"]


def test_side_effect_failure_is_never_automatically_replanned() -> None:
    registry = MalformedRegistry()
    task = SkillTask(
        task_id="refund_ticket",
        intent="create_refund_review_ticket",
        tool_name="create_refund_review_ticket",
        capability="create_refund_review_ticket",
        arguments={"order_id": "R-1", "reason": "late"},
        requires_confirmation=True,
    )
    plan = TaskPlan(tasks=[task], requires_confirmation=True)
    replans: list[str] = []

    async def replan(task: SkillTask, error: str) -> TaskPlan:
        replans.append(error)
        return TaskPlan(tasks=[task])

    report = asyncio.run(MultiSkillExecutor(registry).execute(
        plan,
        request_id="side-effect-no-replan",
        confirmed=True,
        max_replans=1,
        replan=replan,
    ))

    assert report.status == "failed"
    assert report.results[0].failure_code == "invalid_result"
    assert replans == []
    assert registry.calls == ["create_refund_review_ticket"]


def test_planner_replan_prompt_contains_failure_context() -> None:
    llm = MockLLM([
        '{"tasks":[{"task_id":"status","intent":"get_order_status","tool_name":"get_order_status","capability":"get_order_status","arguments":{"order_id":"A"},"dependencies":[]}],"confidence":0.95}'
    ])
    state = CareState(user_id="u1", intent="order", messages=[{"role": "user", "content": "Where is order A?"}])
    task = SkillTask(task_id="status", intent="get_order_status", tool_name="get_order_status", capability="get_order_status", arguments={"order_id": "A"})

    decision = asyncio.run(MultiSkillPlanner(llm).replan(state, task, "target_mismatch: wrong order"))

    assert decision.plan is not None
    prompt = llm.calls[0][0]["content"]
    assert "失败工具=get_order_status" in prompt
    assert "target_mismatch" in prompt


def test_multi_skill_path_is_reachable_from_customer_care_graph() -> None:
    llm = MockLLM([
        '{"intent":"order","entities":{"order_id":"A"},"emotion":"neutral","confidence":0.95}',
        '{"tasks":[{"task_id":"status","intent":"get_order_status","tool_name":"get_order_status","capability":"get_order_status","arguments":{"order_id":"A"},"dependencies":[]},{"task_id":"receipt","intent":"get_order_receipt","tool_name":"get_order_receipt","capability":"get_order_receipt","arguments":{"order_id":"A"},"dependencies":[]}],"rationale":"两个独立查询","confidence":0.95}',
        "已完成订单状态和收据查询。",
    ])
    state = CareState(user_id="u1", messages=[{"role":"user", "content":"Check order A and show my receipt"}])

    result = CareState.model_validate(asyncio.run(build_customer_care_graph(llm).ainvoke(state.model_dump())))

    assert result.multi_skill_requested is True
    assert result.execution_status == "succeeded"
    assert {item["task_id"] for item in result.task_results} == {"status", "receipt"}


def test_graph_replans_mismatched_read_result_and_keeps_parallel_result() -> None:
    catalog = ToolCatalog([
        tool.model_copy(update={"capability": "order_lookup", "task_intents": ["order_lookup"]})
        if tool.name in {"get_order_status", "get_order_details"}
        else tool
        for tool in ToolCatalog.load_default().tools
    ])
    llm = MockLLM([
        '{"intent":"order","entities":{"order_id":"A"},"emotion":"neutral","confidence":0.95}',
        '{"tasks":[{"task_id":"status","intent":"order_lookup","tool_name":"get_order_status","capability":"order_lookup","arguments":{"order_id":"A"},"dependencies":[]},{"task_id":"details","intent":"order_lookup","tool_name":"get_order_details","capability":"order_lookup","arguments":{"order_id":"A"},"dependencies":[]}],"confidence":0.95}',
        '{"tasks":[{"task_id":"status","intent":"order_lookup","tool_name":"get_order_details","capability":"order_lookup","arguments":{"order_id":"A"},"dependencies":[]}],"confidence":0.95}',
        "订单查询已完成。",
    ])
    registry = WrongTargetRegistry(catalog=catalog)
    graph = build_customer_care_graph(
        llm,
        planner=MultiSkillPlanner(llm, catalog),
        executor=MultiSkillExecutor(registry, catalog),
    )
    state = CareState(user_id="u1", messages=[{"role": "user", "content": "Check order A and show my details"}])

    result = CareState.model_validate(asyncio.run(graph.ainvoke(state.model_dump())))

    assert result.execution_status == "succeeded"
    assert result.replan_count == 1
    assert {item["task_id"] for item in result.task_results} == {"status", "details"}
    assert registry.calls.count("get_order_details") == 2
