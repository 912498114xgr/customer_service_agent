"""多 Skill DAG 执行器：并行、依赖、幂等、失败隔离和确认门禁。"""

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable
from typing import Any

from customer_care.orchestration.models import ExecutionReport, SkillTask, TaskPlan, TaskResult
from customer_care.tools.registry import ToolCatalog, ToolRegistry

_LOGGER = logging.getLogger(__name__)


class MultiSkillExecutor:
    """执行经过治理的 DAG；默认只允许本地 ToolRegistry。"""

    def __init__(self, registry: ToolRegistry | None = None, catalog: ToolCatalog | None = None) -> None:
        self._registry = registry or ToolRegistry(catalog)
        self._catalog = catalog or self._registry.catalog
        self._cache: dict[str, TaskResult] = {}

    async def execute(
        self,
        plan: TaskPlan,
        *,
        request_id: str,
        confirmed: bool = False,
        max_replans: int = 0,
        replan: Callable[[SkillTask, str], Any] | None = None,
    ) -> ExecutionReport:
        """执行计划；副作用未确认时只返回 awaiting_confirmation，不执行工具。"""
        if plan.requires_confirmation and not confirmed:
            return ExecutionReport(
                status="awaiting_confirmation",
                results=[
                    TaskResult(
                        task_id=task.task_id,
                        tool_name=task.tool_name,
                        status="awaiting_confirmation",
                        idempotency_key=self._key(request_id, task),
                    )
                    for task in plan.tasks
                    if task.requires_confirmation
                ],
            )

        results: dict[str, TaskResult] = {}
        remaining = {task.task_id: task for task in plan.tasks}
        replan_count = 0
        errors: list[str] = []
        while remaining:
            ready = [
                task
                for task in remaining.values()
                if all(dependency in results and results[dependency].status in {"succeeded", "cached"} for dependency in task.dependencies)
            ]
            blocked = [
                task
                for task in remaining.values()
                if any(dependency in results and results[dependency].status not in {"succeeded", "cached"} for dependency in task.dependencies)
            ]
            for task in blocked:
                results[task.task_id] = self._skipped(task, request_id, "依赖任务失败")
                remaining.pop(task.task_id)
            if not ready:
                if remaining:
                    errors.append("计划无法继续：存在未满足的依赖")
                break
            batch = await asyncio.gather(*(self._run_task(task, request_id) for task in ready))
            for task, result in zip(ready, batch):
                results[task.task_id] = result
                remaining.pop(task.task_id)
                if result.status == "failed":
                    errors.append(result.error or f"任务 {task.task_id} 失败")
                    if replan and replan_count < max_replans:
                        replan_count += 1
                        try:
                            new_plan = await replan(task, result.error or "未知错误")
                            if isinstance(new_plan, TaskPlan):
                                return await self.execute(
                                    new_plan,
                                    request_id=request_id,
                                    confirmed=confirmed,
                                    max_replans=max_replans - replan_count,
                                    replan=replan,
                                )
                        except Exception as exc:
                            errors.append(f"重规划失败：{exc}")

        ordered = [results[task.task_id] for task in plan.tasks if task.task_id in results]
        status = "succeeded" if ordered and all(item.status in {"succeeded", "cached"} for item in ordered) else "partial"
        if ordered and all(item.status == "cached" for item in ordered):
            status = "cached"
        if not ordered or all(item.status == "failed" for item in ordered):
            status = "failed"
        return ExecutionReport(
            status=status,
            results=ordered,
            replan_count=replan_count,
            executed_task_ids=[item.task_id for item in ordered if item.status in {"succeeded", "cached"}],
            skipped_task_ids=[item.task_id for item in ordered if item.status == "skipped"],
            errors=errors,
        )

    async def _run_task(self, task: SkillTask, request_id: str) -> TaskResult:
        key = self._key(request_id, task)
        cached = self._cache.get(key)
        if cached:
            return cached.model_copy(update={"status": "cached"})
        try:
            self._validate_arguments(task)
        except ValueError as exc:
            return TaskResult(task_id=task.task_id, tool_name=task.tool_name, status="failed", error=str(exc), idempotency_key=key)

        attempts = 0
        while attempts < task.max_attempts:
            attempts += 1
            try:
                output = await asyncio.to_thread(self._registry.execute, task.tool_name, task.arguments)
                if not isinstance(output, dict):
                    raise ValueError("工具结果必须是对象")
                result = TaskResult(task_id=task.task_id, tool_name=task.tool_name, status="succeeded", output=output, attempts=attempts, idempotency_key=key)
                self._cache[key] = result
                return result
            except Exception as exc:
                if attempts >= task.max_attempts:
                    _LOGGER.warning("Skill 执行失败 task=%s error=%s", task.task_id, str(exc)[:200])
                    return TaskResult(task_id=task.task_id, tool_name=task.tool_name, status="failed", error=str(exc)[:200], attempts=attempts, idempotency_key=key)
        raise AssertionError("unreachable")

    def _validate_arguments(self, task: SkillTask) -> None:
        tool = self._catalog.get(task.tool_name)
        invalid = set(task.arguments) - set(tool.argument_names)
        if invalid:
            raise ValueError(f"任务 {task.task_id} 参数未被工具声明：{sorted(invalid)}")
        missing = {name for name in tool.argument_names if name.endswith("_id") and name not in task.arguments}
        if missing:
            raise ValueError(f"任务 {task.task_id} 缺少关键参数：{sorted(missing)}")

    @staticmethod
    def _key(request_id: str, task: SkillTask) -> str:
        payload = json.dumps({"request_id": request_id, "task_id": task.task_id, "tool": task.tool_name, "arguments": task.arguments}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _skipped(task: SkillTask, request_id: str, reason: str) -> TaskResult:
        return TaskResult(task_id=task.task_id, tool_name=task.tool_name, status="skipped", error=reason, idempotency_key=MultiSkillExecutor._key(request_id, task))
