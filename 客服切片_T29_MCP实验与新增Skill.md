# 客服切片 T29：MCP 接入实验与新增 Skill 流程

## 这一步在研发流程里的位置

当前 Agent 已经有 Skill 目录、意图/工具路由和 DAG 执行器，但之前的工具都是由进程内 `ToolRegistry` 返回本地固定结果。T29 把“工具如何跨进程调用”替换为真实 MCP 协议：订单只读 MCP Server 独立运行，Agent 作为 MCP Client 发现并调用它发布的工具。

这不是把整个 Agent 改成 MCP。**Agent 仍决定要不要调用哪个 Skill；MCP 负责客户端与工具服务之间的发现和调用协议。**风险门禁、参数约束、任务依赖、超时、结果校验和人工确认仍由现有主程序负责。

## 请求实际经过的路径

```text
用户请求
  -> 意图识别 / Skill Router 或 Planner
  -> ToolCatalog 校验 Skill 名称、参数与风险
  -> MultiSkillExecutor 调用 MCPToolRegistry
  -> MCP Client 启动 stdio 子进程并完成 initialize
  -> tools/list 检查 Server 是否发布该 Skill
  -> tools/call 发送名称和参数
  -> MCP Server 执行只读本地演示函数
  -> Client 将结构化结果交回 Executor
  -> 结果契约校验并写入任务轨迹
```

此实现每次工具调用都会启动一个 MCP 子进程，并执行一次 `tools/list` 再 `tools/call`。这便于观察协议生命周期、隔离 server 崩溃，但会增加启动和发现开销；生产客户端通常会复用生命周期受控的连接、缓存工具元数据并显式管理 server 健康状态。本阶段不把教学型每次启动方式包装成高吞吐实现。

## 如何运行和观察

`uv add "mcp<2"` 将 MCP Python SDK 锁定在 1.x API。2.x SDK 改动了 FastMCP 接口，因此版本上限是当前示例代码的兼容性约束。

在 `.env` 设置：

```dotenv
CUSTOMER_CARE_USE_MCP_ORDER_TOOLS=true
```

然后运行 CLI：

```powershell
uv run --project . python -m customer_care.console
```

如需自由自然语言路由，仍需要配置真实 LLM；那会产生可能的模型费用。本阶段不自动调用模型。无需模型即可跑 MCP 与执行器的端到端验证：

```powershell
uv run --project . --no-sync python -m pytest tests/test_mcp_tools.py -q
```

当前终端中观察到真实 MCP Client 日志 `ListToolsRequest` 和 `CallToolRequest`，测试收到 `{ "order_id": "A-17", "status": "delivered" }`，随后通过已有工具结果契约。测试运行器应与项目虚拟环境一致；这台机器上裸 `uv run pytest` 曾拾取到 `F:\code\agent-rebuild\.venv`，其中没有 MCP SDK，因此推荐显式项目路径并通过 `python -m pytest` 启动。

## 在现有流程里新增一个 Skill

建议下一轮由你亲手做：给订单 MCP Server 增加 `get_order_total`。它只返回演示数据，不调用外部系统。先不要一次性改所有地方；每一步改完运行一条小验证，观察目录/执行器如何约束它。

1. **定义服务端能力**：在 `customer_care/mcp_server.py` 写带类型标注和中文说明的函数，并用 `@server.tool()` 发布。类型注解会参与生成 MCP input schema；业务输入检查仍应在服务端函数内做。
2. **确认协议发现**：启动客户端 `initialize` 后调用 `tools/list`，检查新工具的名称、描述和 input schema 是否符合预期。MCP Server 发布了工具，不代表 Agent 会自动把它路由给用户请求。
3. **加入 Agent 目录**：在 `customer_care/data/tools.json` 增加描述、触发词、正反例和允许参数；这部分为模型候选召回/选择提供业务语义，也声明工具的风险边界。
4. **打通执行映射**：将工具名加入 `MCPToolRegistry._MCP_TOOLS`，让执行器把它转发给 MCP；否则它只会落到本地演示执行器。
5. **补结果契约**：在 `customer_care/orchestration/result_validation.py` 为结果字段、值域和请求目标绑定增加约束。MCP 协议验证消息结构，不会自动证明业务结果正确。
6. **验证成功和失败**：写一条执行器端到端实验，再故意让 Server 返回错误字段/错误目标 ID，确认任务失败且不会被当成成功回答。
7. **写阶段记录**：记录变动文件、实验输入和输出、发现的失败、非目标与下一步；验证后再单独提交。

面试复盘放在实验之后：这次实验实际证明了 MCP 的工具发现和跨进程调用，不证明路由准确率或生产可用性。面试追问时可以拿工具列表、请求/响应轨迹、失败注入和运行成本讨论，不需要把 MCP 协议背成抽象定义。

## 文件变更与原因

| 文件 | 改动 | 原因 |
|---|---|---|
| `pyproject.toml`、`uv.lock` | 增加并锁定 MCP Python SDK 1.x | 将 MCP 作为可复现的项目依赖管理 |
| `customer_care/mcp_server.py` | 新增 stdio MCP Server，发布订单和退款只读查询 | 提供一个真正独立的工具服务进程 |
| `customer_care/tools/registry.py` | 增加 MCP Client Registry，执行 initialize/list/call 并解析结果 | 在不重写 Agent 路由/执行器的情况下接入 MCP |
| `customer_care/service.py` | 支持注入工具 Registry | 允许运行时选择本地或 MCP 执行适配器 |
| `common/config.py`、`customer_care/api.py`、`customer_care/console.py` | 增加 `CUSTOMER_CARE_USE_MCP_ORDER_TOOLS` 开关 | 本地可开关对比，默认行为不变 |
| `tests/test_mcp_tools.py` | 端到端调用 MCP Server 并验证执行器结果契约 | 覆盖真实协议而非模拟 MCP Client |
| `README.md` | 增加运行说明和项目解释器测试命令 | 降低首次体验的环境歧义 |

## 已验证与尚未验证

- MCP SDK 安装与 uv 锁定完成，SDK 限制为 `<2`。
- 真实本地 MCP stdio Client/Server 执行 `initialize`、`tools/list`、`tools/call`，执行器获得结果且通过已有结果契约。
- 第二条验证确认未接入的政策 Skill 仍走原本地演示 Registry。
- 未接真实模型；未连真实订单 API；未压测连接复用、并发和延迟；未实现 MCP Server 授权、审计和远程 HTTP 传输。
- 当前工具实现包含在 Agent 项目包内，目的是学习协议与研发接入点。真正的服务边界可将 Server 移到独立仓库/部署单元，再配置远程 MCP transport 和身份认证。
