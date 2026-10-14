# 客服切片 T28：SQLite 持久化与原子确认

## 阶段目标

T26 的确认 token 和会话状态只保存在单进程内存。服务重启后会丢失待确认操作；两个服务实例同时处理同一 token 时，Python 内部锁也不能提供共享互斥。本阶段增加可本地运行的 SQLite 快照后端，并把确认/拒绝动作改为存储层原子状态迁移。

本阶段不接真实退款、支付、订单等下游，不实现 LangGraph checkpoint、执行中断续跑、外部审批和分布式事务。数据库只保证一个确认 token 在状态库中最多被认领一次；数据库提交和外部副作用不可能由当前本地实现组成同一事务，因此不承诺 exactly-once。

## 设计决策

- 增加 `MemoryStore.claim_confirmation` 契约：存储层对当前状态、confirmation ID、token 摘要和过期时间做一次原子检查，并持久化 `executing` 或 `declined` 状态。
- SQLite 使用标准库 `sqlite3` 与 `BEGIN IMMEDIATE`，无需新增运行依赖，适合单机开发/演练。IO 通过 `asyncio.to_thread` 执行，避免同步数据库操作直接阻塞事件循环。
- Redis 使用 WATCH/MULTI 乐观事务；内存存储使用现有 per-user 锁。服务层不再把 Python `asyncio.Lock` 当作跨进程正确性的依据。
- SQLite 通过 `CUSTOMER_CARE_SQLITE_PATH` 显式启用；未配置时保留原有内存后端，避免静默创建持久文件。
- 相比直接将 API 默认切换为 SQLite，本阶段采用 opt-in 配置，减少对既有本地体验、测试与部署行为的隐式改变。

## 文件变更记录

| 文件 | 改动 | 原因 |
|---|---|---|
| `customer_care/memory/base.py` | 增加原子确认认领存储契约 | 将一次性 token 的竞争语义放到有能力执行原子操作的边界 |
| `customer_care/memory/confirmation.py` | 集中实现 token 校验、过期、确认和拒绝迁移规则 | 避免不同存储后端对状态迁移有行为偏差 |
| `customer_care/memory/in_memory.py` | 在用户锁内原子迁移确认状态 | 保持原有 Mock/测试路径与新契约一致 |
| `customer_care/memory/redis_memory.py` | 以 WATCH/MULTI 更新 current、sequence 和版本快照 | 使已有 Redis 适配器具备共享存储级的原子认领逻辑；本阶段未连接真实 Redis 验证 |
| `customer_care/memory/sqlite_memory.py` | 新增 SQLite 不可变快照、当前状态、回滚和事务认领实现 | 提供免外部服务、重启后可恢复的本地存储后端 |
| `customer_care/memory/__init__.py` | 导出 SQLite 后端 | 统一存储模块入口 |
| `customer_care/service.py` | 确认和拒绝都先请求存储层原子认领，再进入后续处理 | 防止多个服务实例读取相同 pending 状态后各自执行 |
| `common/config.py`、`customer_care/api.py`、`customer_care/console.py` | 支持 `CUSTOMER_CARE_SQLITE_PATH` 配置 | 让 CLI/API 可以直接体验持久状态，不需改业务代码 |
| `tests/test_sqlite_memory.py`、`tests/test_demo.py` | 覆盖持久化、回滚和两个独立服务实例的并发认领 | 验证存储语义及服务层集成，不把单测替代生产验证 |
| `.gitignore` | 忽略本地 SQLite 数据库文件 | 防止会话数据误入版本库 |
| `README.md` | 补充启用方式与事务边界 | 让运行者知道默认后端和持久化能力的真实范围 |
| `项目最终验收标准_压力面试矩阵.md` | 更新副作用原子性及恢复证据 | 保持矩阵状态与新代码/实验一致 |
| `VibeCoding协作契约.md` | 更新阶段队列和真实模型评测授权边界 | 对齐当前工作状态与下一步安排 |

## 运行方式

在项目 `.env` 增加：

```dotenv
CUSTOMER_CARE_SQLITE_PATH=./data/customer-care.db
```

然后沿用原入口：

```powershell
uv run python -m customer_care.console
uv run uvicorn customer_care.api:create_app --factory --reload
```

关闭 SQLite 时删除该配置，系统恢复为默认的进程内存状态。数据库保存完整会话快照，可能含用户文本；文件应按敏感数据管理，不要提交或上传。

## 验证结果

- `uv run pytest -q`：94 passed，1 skipped。
- `uv run python -m customer_care.self_test`：退出码 0，三项本地自测均通过。
- `git diff --check`：通过。
- SQLite 快照测试关闭并重新创建存储对象后仍能读取状态，并可回滚到历史版本。
- 两个分别构造的 `CustomerCareService` 共享同一 SQLite 文件并同时确认同一 token：一个成功完成，一个收到冲突；最终状态只有一份已完成任务结果。
- 未运行真实 Redis 集成、操作系统多进程压力测试或真实业务下游；上述结果不证明生产容量和 exactly-once。

## 面试追问口径

**两个 worker 同时收到确认请求怎么办？** 两边可以并发读取，但认领必须进入存储层事务。SQLite `BEGIN IMMEDIATE` 串行化写事务，后到请求重新读到的状态已不再是 pending，因此无法消费同一个 token。Redis 走 WATCH/MULTI 竞争版本；冲突方重读后再判定状态。当前测试覆盖两个独立服务实例，不是多进程负载测试。

**确认记录是 executing，进程却崩溃了怎么办？** 系统不会自动重试，因为外部下游可能已经执行。状态可以在重启后读到，但当前没有恢复 worker 或下游查询/幂等协议，因此要标记为人工 reconciliation。仅有状态持久化不等于执行可恢复。

**这是否保证不会重复退款？** 不保证。它只保证同一个确认凭证在状态库里最多被一个服务实例认领。生产上还要将稳定幂等键传给下游、保存调用意图和下游回执，并定义超时后的查询/补偿流程；下游不支持幂等时仍存在结果未知窗口。

## 后续缺口

- SQLite 适合单机演练，不是多节点部署推荐数据库；需要真实 Redis/关系数据库集成与并发压测。
- 当前 `executing` 状态没有自动恢复、租约或人工核对工作台。
- 还没有工具级持久化幂等记录、审计事件和下游回执。
- T27 真实模型评测器尚未获得用户本地授权并运行，模型路由效果仍未知。
