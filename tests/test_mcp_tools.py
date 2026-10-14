import asyncio

from customer_care.orchestration.executor import MultiSkillExecutor
from customer_care.orchestration.models import SkillTask, TaskPlan
from customer_care.tools.registry import MCPToolRegistry


def test_mcp_server_exposes_order_tools_and_executor_validates_result() -> None:
    registry = MCPToolRegistry()
    plan = TaskPlan(tasks=[
        SkillTask(
            task_id="lookup",
            intent="get_order_status",
            tool_name="get_order_status",
            capability="get_order_status",
            arguments={"order_id": "A-17"},
        )
    ])

    report = asyncio.run(MultiSkillExecutor(registry).execute(plan, request_id="mcp-e2e"))

    assert report.status == "succeeded"
    assert report.results[0].output == {"order_id": "A-17", "status": "delivered"}


def test_mcp_registry_keeps_unconfigured_skills_local() -> None:
    result = MCPToolRegistry().execute("get_refund_policy", {})

    assert result["money_moved"] is False
    assert "Eligible refund" in result["policy"]
