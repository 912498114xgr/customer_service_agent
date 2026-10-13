"""多 Skill 编排使用的领域模型。"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SkillTask(BaseModel):
    """计划中的一个 Skill 任务。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1, max_length=64)
    intent: str = Field(min_length=1, max_length=64)
    tool_name: str = Field(min_length=1, max_length=128)
    arguments: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "low"
    requires_confirmation: bool = False
    max_attempts: int = Field(default=1, ge=1, le=3)


class TaskPlan(BaseModel):
    """可执行的有向无环 Skill 计划。"""

    model_config = ConfigDict(extra="forbid")

    tasks: list[SkillTask] = Field(min_length=1, max_length=8)
    requires_confirmation: bool = False
    rationale: str = ""

    @model_validator(mode="after")
    def validate_graph(self) -> "TaskPlan":
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("任务 ID 必须唯一")
        known = set(task_ids)
        for task in self.tasks:
            missing = set(task.dependencies) - known
            if missing:
                raise ValueError(f"任务 {task.task_id} 依赖不存在的任务：{sorted(missing)}")
            if task.task_id in task.dependencies:
                raise ValueError(f"任务 {task.task_id} 不能依赖自己")
        if self._has_cycle():
            raise ValueError("Skill 计划不能包含循环依赖")
        if any(task.requires_confirmation for task in self.tasks) and not self.requires_confirmation:
            self.requires_confirmation = True
        return self

    def _has_cycle(self) -> bool:
        dependencies = {task.task_id: set(task.dependencies) for task in self.tasks}
        resolved: set[str] = set()
        while dependencies:
            ready = {task_id for task_id, deps in dependencies.items() if not deps or deps <= resolved}
            if not ready:
                return True
            resolved.update(ready)
            for task_id in ready:
                dependencies.pop(task_id)
        return False


class TaskResult(BaseModel):
    """单个 Skill 的执行结果和可重放信息。"""

    task_id: str
    tool_name: str
    status: Literal["succeeded", "failed", "skipped", "awaiting_confirmation", "cached"]
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    attempts: int = 0
    idempotency_key: str


class ExecutionReport(BaseModel):
    """整个计划的执行报告。"""

    status: Literal["succeeded", "partial", "failed", "awaiting_confirmation", "cached"]
    results: list[TaskResult] = Field(default_factory=list)
    replan_count: int = 0
    executed_task_ids: list[str] = Field(default_factory=list)
    skipped_task_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """转换为 API/trace 可直接返回的结构。"""
        return self.model_dump(mode="json")
