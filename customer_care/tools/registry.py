"""标准化客服工具目录及无副作用的本地执行器。"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    category: str
    description: str
    trigger_keywords: list[str]
    pos_examples: list[str]
    neg_examples: list[str]
    argument_names: list[str]
    side_effect: bool = False
    requires_confirmation: bool = False
    idempotency_required: bool = True
    capability: str | None = None
    task_intents: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def set_default_capability(self) -> "ToolDefinition":
        if self.capability is None:
            self.capability = self.name
        if not self.task_intents:
            self.task_intents = [self.capability]
        return self


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


class MCPToolRegistry(ToolRegistry):
    """将订单只读 Skill 通过 stdio MCP 转发，其余工具保留本地演示执行。"""

    _MCP_TOOLS = {"get_order_status", "get_refund_status"}

    def __init__(self, catalog: ToolCatalog | None = None) -> None:
        super().__init__(catalog)

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._MCP_TOOLS:
            return super().execute(name, arguments)
        import asyncio

        return asyncio.run(self._execute_mcp(name, arguments))

    async def _execute_mcp(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        import shutil
        from datetime import timedelta
        from pathlib import Path
        from sysconfig import get_paths

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server = StdioServerParameters(
            command=shutil.which("uv") or "uv",
            args=[
                "run",
                "--project",
                str(Path(__file__).resolve().parents[2]),
                "--no-sync",
                "python",
                "-m",
                "customer_care.mcp_server",
            ],
            cwd=Path(__file__).resolve().parents[2],
        )
        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                if name not in {tool.name for tool in tools.tools}:
                    raise RuntimeError(f"MCP 服务未发布所需 Skill：{name}")
                result = await session.call_tool(
                    name,
                    arguments,
                    read_timeout_seconds=timedelta(seconds=10),
                )
                if result.isError:
                    raise RuntimeError(f"MCP Skill 调用失败：{name}")
                if result.structuredContent is not None:
                    return result.structuredContent
                text_blocks = [item.text for item in result.content if getattr(item, "text", None)]
                if len(text_blocks) != 1:
                    raise RuntimeError(f"MCP Skill 返回格式错误：{name}")
                try:
                    payload = json.loads(text_blocks[0])
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"MCP Skill 未返回 JSON 对象：{name}") from exc
                if not isinstance(payload, dict):
                    raise RuntimeError(f"MCP Skill 返回值必须是 JSON 对象：{name}")
                return payload

