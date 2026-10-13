"""多 Skill DAG 执行器：并行、依赖、幂等、失败隔离和确认门禁。"""

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable
from typing import Any

from customer_care.orchestration.models import ExecutionReport, SkillTask, TaskPlan, TaskResult
from customer_care.orchestration.result_validation import ToolResultError, validate_task_contract, validate_tool_result
from customer_care.tools.registry import ToolCatalog, ToolRegistry

_LOGGER = logging.getLogger(__name__)


class MultiSkillExecutor:
    """执行经过治理的 DAG；默认只允许本地 ToolRegistry。"""

    def __init__(self, registry: ToolRegistry | None = None, catalog: ToolCatalog | None = None) -> None:
        self._registry = registry or ToolRegistry(catalog)
        self._catalog = catalog or self._registry.catalog
        self._cache: dict[str, TaskResult] = {}

    @property
    def cache_size(self) -> int:
        """返回当前幂等缓存条目数量，供观测和测试使用。"""
        return len(self._cache)

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
            failed_batch: list[tuple[SkillTask, TaskResult]] = []
            for task, result in zip(ready, batch):
                results[task.task_id] = result
                remaining.pop(task.task_id)
                if result.status == "failed":
                    errors.append(result.error or f"任务 {task.task_id} 失败")
                    failed_batch.append((task, result))
            for task, result in failed_batch:
                try:
                    tool = self._catalog.get(task.tool_name)
                except ValueError:
                    tool = None
                can_replan = (
                    replan is not None
                    and replan_count < max_replans
                    and not task.requires_confirmation
                    and tool is not None
                    and not tool.side_effect
                    and result.failure_code in {"timeout", "tool_error", "invalid_result", "target_mismatch", "invalid_plan"}
                )
                if not can_replan:
                    continue
                replan_count += 1
                try:
                    new_plan = await replan(task, f"{result.failure_code}: {result.error or '未知错误'}")
                    if not isinstance(new_plan, TaskPlan):
                        continue
                    if (
                        len(new_plan.tasks) != 1
                        or new_plan.tasks[0].task_id != task.task_id
                        or new_plan.tasks[0].intent != task.intent
                        or new_plan.tasks[0].capability != task.capability
                    ):
                        errors.append("重规划被拒绝：只能替换同一任务目标下的单个等价 Skill")
                        continue
                    replanned = await self.execute(
                        new_plan,
                        request_id=request_id,
                        confirmed=confirmed,
                        max_replans=max_replans - replan_count,
                        replan=replan,
                    )
                    combined = {item.task_id: item for item in results.values()}
                    combined.update({item.task_id: item for item in replanned.results})
                    ordered_results = [combined[item.task_id] for item in plan.tasks if item.task_id in combined]
                    original_ids = {task.task_id for task in plan.tasks}
                    ordered_results.extend(item for task_id, item in combined.items() if task_id not in original_ids)
                    statuses = {item.status for item in ordered_results}
                    if statuses and statuses <= {"succeeded", "cached"}:
                        status = "cached" if statuses == {"cached"} else "succeeded"
                    elif statuses and statuses <= {"failed"}:
                        status = "failed"
                    else:
                        status = "partial"
                    return ExecutionReport(
                        status=status,
                        results=ordered_results,
                        replan_count=replan_count + replanned.replan_count,
                        executed_task_ids=[item.task_id for item in ordered_results if item.status in {"succeeded", "cached"}],
                        skipped_task_ids=[item.task_id for item in ordered_results if item.status == "skipped"],
                        errors=errors + replanned.errors,
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
        try:
            validate_task_contract(task, self._catalog)
            self._validate_arguments(task)
        except (ToolResultError, ValueError) as exc:
            return TaskResult(task_id=task.task_id, tool_name=task.tool_name, status="failed", failure_code="invalid_plan", error=str(exc), idempotency_key=key)
        cached = self._cache.get(key)
        if cached:
            return cached.model_copy(update={"status": "cached"})

        attempts = 0
        while attempts < task.max_attempts:
            attempts += 1
            try:
                output = await asyncio.wait_for(
                    asyncio.to_thread(self._registry.execute, task.tool_name, task.arguments),
                    timeout=task.timeout_seconds,
                )
                output = validate_tool_result(task, output, self._catalog)
                result = TaskResult(task_id=task.task_id, tool_name=task.tool_name, status="succeeded", output=output, attempts=attempts, idempotency_key=key)
                self._cache[key] = result
                return result
            except ToolResultError as exc:
                _LOGGER.warning("Skill 结果校验失败 task=%s code=%s", task.task_id, exc.code)
                return TaskResult(task_id=task.task_id, tool_name=task.tool_name, status="failed", failure_code=exc.code, error=str(exc)[:200], attempts=attempts, idempotency_key=key)
            except TimeoutError:
                if attempts >= task.max_attempts:
                    return TaskResult(task_id=task.task_id, tool_name=task.tool_name, status="failed", failure_code="timeout", error="tool timeout", attempts=attempts, idempotency_key=key)
            except Exception as exc:
                if attempts >= task.max_attempts:
                    error = str(exc)[:200]
                    _LOGGER.warning("Skill 执行失败 task=%s error=%s", task.task_id, error)
                    return TaskResult(task_id=task.task_id, tool_name=task.tool_name, status="failed", failure_code="tool_error", error=error, attempts=attempts, idempotency_key=key)
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
