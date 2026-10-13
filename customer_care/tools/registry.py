"""标准化客服工具目录及无副作用的本地执行器。"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    category: str
    description: str
    trigger_keywords: list[str]
    pos_examples: list[str]
    neg_examples: list[str]
    argument_names: list[str]


class ToolCatalog:
    def __init__(self, tools: list[ToolDefinition]) -> None:
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("工具名称必须唯一")
        self._tools = tuple(tools)
        self._by_name = {tool.name: tool for tool in tools}

    @classmethod
    def load_default(cls) -> "ToolCatalog":
        data_path = Path(__file__).resolve().parents[1] / "data" / "tools.json"
        raw_tools = json.loads(data_path.read_text(encoding="utf-8"))
        return cls([ToolDefinition.model_validate(item) for item in raw_tools])

    @property
    def tools(self) -> tuple[ToolDefinition, ...]:
        return self._tools

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise ValueError(f"未知工具：{name}") from exc


class ToolRegistry:
    """只返回本地固定样例，不调用真实订单、支付或账户系统。"""

    def __init__(self, catalog: ToolCatalog | None = None) -> None:
        self.catalog = catalog or ToolCatalog.load_default()

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self.catalog.get(name)
        safe_arguments = {key: value for key, value in arguments.items() if key in tool.argument_names}
        order_id = str(safe_arguments.get("order_id", "DEMO-ORDER"))

        if name == "get_refund_status":
            return {"order_id": order_id, "status": "processing", "estimated_days": 3}
        if name == "get_order_status":
            return {"order_id": order_id, "status": "delivered"}
        if name == "check_refund_eligibility":
            return {"order_id": order_id, "eligible_for_review": True, "money_moved": False}
        if name == "get_refund_policy":
            return {"policy": "Eligible refund requests are reviewed by a human agent.", "money_moved": False}
        if name == "get_delivery_status":
            return {"order_id": order_id, "status": "on_the_way", "eta_minutes": 18}
        return {
            "tool": name,
            "status": "demo_result",
            "arguments": safe_arguments,
            "side_effects": False,
        }

