"""Skill 输出契约与请求目标一致性校验。"""

from typing import Any

from customer_care.orchestration.models import SkillTask
from customer_care.tools.registry import ToolCatalog


class ToolResultError(ValueError):
    """工具输出不满足契约，或没有回答原任务目标。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


_CONTRACTS: dict[str, dict[str, Any]] = {
    "get_order_status": {
        "required": {"order_id": str, "status": str},
        "enums": {"status": {"pending", "preparing", "on_the_way", "delivered", "cancelled"}},
        "bindings": {"order_id": "order_id"},
    },
    "get_refund_status": {
        "required": {"order_id": str, "status": str, "estimated_days": int},
        "enums": {"status": {"pending", "processing", "refunded", "failed"}},
        "minimums": {"estimated_days": 0},
        "bindings": {"order_id": "order_id"},
    },
    "get_delivery_status": {
        "required": {"order_id": str, "status": str, "eta_minutes": int},
        "enums": {"status": {"preparing", "on_the_way", "delivered", "delayed"}},
        "minimums": {"eta_minutes": 0},
        "bindings": {"order_id": "order_id"},
    },
    "check_refund_eligibility": {
        "required": {"order_id": str, "eligible_for_review": bool, "money_moved": bool},
        "literals": {"money_moved": False},
        "bindings": {"order_id": "order_id"},
    },
    "get_refund_policy": {
        "required": {"policy": str, "money_moved": bool},
        "literals": {"money_moved": False},
    },
}

_GENERIC_CONTRACT = {
    "required": {"tool": str, "status": str, "arguments": dict, "side_effects": bool},
    "enums": {"status": {"demo_result"}},
    "literals": {"side_effects": False},
}


def validate_task_contract(task: SkillTask, catalog: ToolCatalog) -> None:
    """拒绝任务声明的能力与实际工具不一致。"""
    definition = catalog.get(task.tool_name)
    if task.intent not in definition.task_intents:
        raise ToolResultError(
            "invalid_plan",
            f"任务目标 {task.intent!r} 不属于工具 {task.tool_name!r} 支持的意图",
        )
    if task.capability != definition.capability:
        raise ToolResultError(
            "invalid_plan",
            f"任务能力 {task.capability!r} 与工具 {task.tool_name!r} 的能力 {definition.capability!r} 不匹配",
        )


def validate_tool_result(task: SkillTask, output: Any, catalog: ToolCatalog) -> dict[str, Any]:
    """校验输出类型、必需字段、值域和请求参数回显。"""
    if not isinstance(output, dict):
        raise ToolResultError("invalid_result", "工具结果必须是对象")
    definition = catalog.get(task.tool_name)
    contract = _CONTRACTS.get(task.tool_name, _GENERIC_CONTRACT)
    for field, expected_type in contract["required"].items():
        value = output.get(field)
        if field not in output or not _matches_type(value, expected_type):
            raise ToolResultError("invalid_result", f"工具 {task.tool_name} 结果字段 {field} 缺失或类型错误")
    for field, allowed in contract.get("enums", {}).items():
        if output[field] not in allowed:
            raise ToolResultError("invalid_result", f"工具 {task.tool_name} 返回未知 {field} 值")
    for field, expected in contract.get("literals", {}).items():
        if output[field] != expected:
            raise ToolResultError("invalid_result", f"工具 {task.tool_name} 结果字段 {field} 不满足安全约束")
    for field, minimum in contract.get("minimums", {}).items():
        if output[field] < minimum:
            raise ToolResultError("invalid_result", f"工具 {task.tool_name} 结果字段 {field} 不满足范围约束")
    for output_field, argument_name in contract.get("bindings", {}).items():
        expected = task.arguments.get(argument_name)
        if output[output_field] != expected:
            raise ToolResultError(
                "target_mismatch",
                f"工具 {task.tool_name} 返回目标 {output[output_field]!r}，与请求目标 {expected!r} 不一致",
            )
    if contract is _GENERIC_CONTRACT:
        if output["tool"] != task.tool_name:
            raise ToolResultError("target_mismatch", "工具响应中的 tool 与实际调用不一致")
        if output["arguments"] != task.arguments:
            raise ToolResultError("target_mismatch", "工具响应参数与请求参数不一致")
    return output


def _matches_type(value: Any, expected_type: type) -> bool:
    if expected_type is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected_type)
