import asyncio
import time

import pytest

from customer_care.orchestration import MultiSkillExecutor, SkillTask, TaskPlan
from customer_care.orchestration import MultiSkillPlanner
from common.mock_llm import MockLLM
from customer_care.state import CareState
from customer_care.tools.registry import ToolRegistry


class RecordingRegistry(ToolRegistry):
    def __init__(self, *, fail_tools: set[str] | None = None, delay: float = 0.0) -> None:
        super().__init__()
        self.calls: list[str] = []
        self.fail_tools = fail_tools or set()
        self.delay = delay

    def execute(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        if self.delay:
            time.sleep(self.delay)
        self.calls.append(name)
        if name in self.fail_tools:
            raise RuntimeError(f"injected failure: {name}")
        return {"tool": name, "status": "ok", "arguments": arguments}


def test_independent_read_tasks_run_in_parallel() -> None:
    registry = RecordingRegistry(delay=0.1)
    plan = TaskPlan(tasks=[
        SkillTask(task_id="order", intent="order", tool_name="get_order_status", arguments={"order_id": "A"}),
        SkillTask(task_id="coupon", intent="coupon", tool_name="get_coupon_status", arguments={"coupon_id": "C"}),
    ])

    started = time.perf_counter()
    report = asyncio.run(MultiSkillExecutor(registry).execute(plan, request_id="r1"))
    elapsed = time.perf_counter() - started

    assert report.status == "succeeded"
    assert set(registry.calls) == {"get_order_status", "get_coupon_status"}
    assert elapsed < 0.19


def test_side_effect_task_requires_confirmation_before_execution() -> None:
    registry = RecordingRegistry()
    plan = TaskPlan(tasks=[SkillTask(task_id="refund", intent="refund", tool_name="create_refund_review_ticket", arguments={"order_id": "R-1", "reason": "late"}, requires_confirmation=True)])

    report = asyncio.run(MultiSkillExecutor(registry).execute(plan, request_id="r2"))

    assert report.status == "awaiting_confirmation"
    assert registry.calls == []


def test_failed_dependency_skips_downstream_task() -> None:
    registry = RecordingRegistry(fail_tools={"get_order_status"})
    plan = TaskPlan(tasks=[
        SkillTask(task_id="status", intent="order", tool_name="get_order_status", arguments={"order_id": "A"}),
        SkillTask(task_id="refund", intent="refund", tool_name="check_refund_eligibility", arguments={"order_id": "A"}, dependencies=["status"]),
    ])

    report = asyncio.run(MultiSkillExecutor(registry).execute(plan, request_id="r3"))

    assert report.status == "partial"
    assert {item.task_id: item.status for item in report.results} == {"status": "failed", "refund": "skipped"}


def test_same_request_is_idempotent() -> None:
    registry = RecordingRegistry()
    executor = MultiSkillExecutor(registry)
    plan = TaskPlan(tasks=[SkillTask(task_id="status", intent="order", tool_name="get_order_status", arguments={"order_id": "A"})])

    first = asyncio.run(executor.execute(plan, request_id="r4"))
    second = asyncio.run(executor.execute(plan, request_id="r4"))

    assert first.status == "succeeded"
    assert second.status == "cached"
    assert registry.calls == ["get_order_status"]


def test_cycle_is_rejected() -> None:
    with pytest.raises(ValueError, match="循环依赖"):
        TaskPlan(tasks=[
            SkillTask(task_id="a", intent="order", tool_name="get_order_status", dependencies=["b"]),
            SkillTask(task_id="b", intent="order", tool_name="get_order_details", dependencies=["a"]),
        ])


def test_planner_marks_high_risk_side_effect() -> None:
    llm = MockLLM([
        '{"tasks":[{"task_id":"status","intent":"refund","tool_name":"create_refund_review_ticket","arguments":{"order_id":"R-1","reason":"late"},"dependencies":[]}],"rationale":"需要人工审核","confidence":0.95}'
    ])
    state = CareState(user_id="u1", intent="refund", entities={"order_id": "R-1"}, messages=[{"role": "user", "content": "请审核退款"}])

    decision = asyncio.run(MultiSkillPlanner(llm).plan(state))

    assert decision.plan is not None
    assert decision.plan.requires_confirmation is True
    assert decision.plan.tasks[0].risk_level == "high"
