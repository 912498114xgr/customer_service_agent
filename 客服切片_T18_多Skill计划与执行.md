# 客服切片：T18 多 Skill 计划与执行

## 解决的问题

以前的路由只有一个 `selected_tool`，无法解释“一句话触发两个 Skill”“一个任务依赖另一个任务”“Skill 选错后如何恢复”。T18 将请求建模为 `TaskPlan -> SkillTask -> TaskResult`，并提供 DAG 执行器。

## 核心机制

- 计划校验：任务 ID 唯一、依赖存在、拒绝循环依赖。
- 并行执行：依赖已满足的独立任务使用 `asyncio.gather` 同批执行。
- 串行执行：有依赖的任务必须等待上游成功或缓存成功。
- 失败隔离：上游失败时下游标记 `skipped`，不会盲目执行。
- 副作用确认：有 `requires_confirmation` 的 Skill 在确认前只返回 `awaiting_confirmation`。
- 参数治理：工具声明外参数拒绝，关键 `_id` 参数缺失时不执行。
- 幂等：请求 ID、任务 ID、工具名和参数生成幂等键；同一执行器重放返回 `cached`。
- 有限重规划接口：失败任务可回调一次替代计划，调用者可限制 `max_replans`，避免无限循环。

## 面试问题对应

**一句话触发两个 Skill**：拆成多个任务；无依赖的查询并行，有依赖的任务按 DAG 串行。

**Skill 选错**：执行前做工具类别和参数校验，执行后检查结果；失败只能有限重规划，仍失败则部分失败/人工兜底。

**副作用怎么办**：Skill 元数据声明风险、确认和幂等要求；确认前不调用工具。

## 验证

```powershell
uv --cache-dir .uv-cache run --no-sync pytest -q tests/test_orchestration.py
```

覆盖并行耗时、确认门禁、依赖失败、循环依赖、重放幂等和规划器风险标记，共 6 项测试。

当前仍是本地工具执行器；真实订单/支付系统还需要把 `ToolRegistry` 替换成带超时、鉴权、审计和结果 schema 的适配器。
