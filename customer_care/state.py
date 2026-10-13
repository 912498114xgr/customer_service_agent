"""客服对话流程中传递的状态。"""

from typing import Any

from pydantic import BaseModel, Field


class CareState(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)
    user_id: str
    language: str = "en"
    intent: str | None = None
    intent_confidence: float = 0.0
    entities: dict[str, Any] = Field(default_factory=dict)
    rewritten_query: str | None = None
    rewrite_ambiguous: bool = False
    rewrite_explanation: str | None = None
    route: str | None = None
    selected_tool: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    tool_result: dict[str, Any] = Field(default_factory=dict)
    task_plan: dict[str, Any] | None = None
    task_results: list[dict[str, Any]] = Field(default_factory=list)
    execution_status: str | None = None
    awaiting_confirmation: bool = False
    replan_count: int = 0
    multi_skill_requested: bool = False
    handoff_summary: str | None = None
    context_summary: str | None = None
    knowledge_context: str = ""
    knowledge_sources: list[dict[str, Any]] = Field(default_factory=list)
    answer: str | None = None
    emotion: str | None = None

