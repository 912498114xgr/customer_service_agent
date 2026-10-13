# Agent Rebuild

用 Python + LangGraph 重实现两个本地可验证的 Agent 垂直切片。当前客服切片推进到 T22；诊断切片尚未开始。T18-T22 加入本地多 Skill 计划/DAG 执行、主图接入、结果契约、只读有限恢复和多任务评测基线；多任务入口仍是启发式，不能视作生产级路由。

## 环境准备

要求 Python 3.11+ 和 uv。首次安装依赖：

```powershell
uv sync --extra dev
```

`pyproject.toml` 声明项目依赖，`uv.lock` 固定解析结果，`.venv` 是项目隔离环境。后续新增运行依赖用 `uv add 包名`，新增测试/开发依赖用 `uv add --dev 包名`。运行命令统一通过 `uv run`，不需要手动激活虚拟环境。

## 测试与离线自测

```powershell
uv run pytest -q
uv run python -m customer_care.self_test
```

自测覆盖退款咨询端到端链路、规则质检和理赔金额上限。默认使用离线 LLM Mock 和本地固定工具结果，不访问外部模型、Redis、订单或支付系统。

服务化体验需要额外 Web 依赖：

```powershell
uv sync --extra web
uv run uvicorn customer_care.api:create_app --factory --reload
```

API 端点为 `/health`、`POST /v1/chat` 和 `/v1/chat/{user_id}/stream`；工作台文件在 `customer_care/web/index.html`。当前环境的网络权限未完成 FastAPI/uvicorn 下载，核心离线测试不依赖它们。

## 路由评测基线

T9 新增了 40 条人工标注的工具路由样本，覆盖 30 个工具、三种语言和近义混淆。运行：

```powershell
uv run python -m customer_care.evaluation
```

当前离线基线为 `Top-1=70.0%`、`Recall@3=92.5%`、`Recall@10=100.0%`。Recall@3 用来衡量第二级模型收到的候选是否包含正确工具，不能当作端到端回答准确率。逐条 bad case 和类别分项结果见 [客服切片_T9_评测基线.md](客服切片_T9_评测基线.md)。

T22 增加了 25 条多 Skill 固定回归样本和计划评分器。当前图中的连接词入口检测基线为 `Accuracy=52.0%`、`Precision=66.7%`、`Recall=50.0%`、`F1=57.1%`；这是小型人工回归集上的启发式指标，不是生产数据或真实模型准确率。完整范围及限制见 [客服切片_T22_多Skill评测基线.md](客服切片_T22_多Skill评测基线.md)。

## 交互式运行

客服切片现在提供一个终端交互入口，可以由你扮演用户输入问题，并查看每轮 Agent 的意图、实体、路由、工具和结果：

```powershell
uv run python -m customer_care.console
```

输入 `/help` 查看命令，输入 `/reset` 重置会话，输入 `/quit` 退出。要体验真实模型，把 `.env` 中的 `USE_MOCK` 设为 `false` 并配置模型 Key；Mock 模式只用于离线演示和测试，不会理解任意自然语言。

## 连接 OpenAI 兼容模型

复制 `.env.example` 为 `.env`，设置：

```dotenv
USE_MOCK=false
LLM_API_KEY=你的密钥
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

也兼容 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`、`OPENAI_TIMEOUT_SECONDS` 和 `OPENAI_MAX_RETRIES` 命名；两套变量同时存在时优先使用 `LLM_*`。API Key 只放在本机 `.env` 或进程环境变量中，不要粘贴到代码、提交记录或聊天中。

不要把真实 `.env` 或密钥提交到版本库。真实 LLM 请求会产生网络和费用；自动化测试默认不启用真实请求。

T10 增加了 `LLM_TIMEOUT_SECONDS` 和 `LLM_MAX_RETRIES`。模型请求失败时，图节点会分别进入澄清、候选回退或本地回复模板，不会因为模型故障执行退款、赔付或账户变更。详细配置和边界见 [客服切片_T10_真实模型接入.md](客服切片_T10_真实模型接入.md)。

## 客服切片结构

```text
common/                         配置、统一 LLM 接口、结构化输出
customer_care/state.py          LangGraph 对话状态
customer_care/graph.py          状态图装配与分支
customer_care/nodes/            语言、意图、改写、澄清、路由、回复节点
customer_care/memory/           内存与 Redis 版本快照
customer_care/tools/            30 个工具目录、两级路由、本地执行器
customer_care/quality/          硬规则和模型评分
customer_care/claims/           规则限额和人工审核建议
customer_care/data/tools.json   五类标准化工具描述
tests/                          离线单测与端到端测试
```

## 安全边界

- 工具执行器只返回本地固定样例，不连接生产订单、支付或账户服务。
- 退款、赔付和豁免只生成建议及待人工审核工单；代码没有资金执行接口。
- 理赔上限由 Python 规则根据可信订单金额计算，LLM 输出不包含可执行金额。
- 质检硬规则命中后总分一票否决，模型分数不能覆盖规则结果。
- 高情绪、低意图置信度或指代歧义会转人工/追问，不强行自助。

## 当前限制与后续工作

- 两级路由用关键词 + 词袋 Jaccard 相似度做轻量召回，未接真实 embedding、Reranker 或 RAG。
- 30 个工具是固定的本地演示数据，不代表真实业务 API。
- Redis 适配器已实现原子保存和回滚脚本，但需要本地 Redis 服务才能做集成验证；当前自动测试不依赖 Redis。
- 语言识别、意图识别和最终回复仍没有真实标注集指标；T10 已接入真实模型客户端，但当前只有工具粗筛的离线路由基线，不能代表端到端 Agent 准确率。
- 图本身未配置 LangGraph checkpointer；多轮持久化由上层调用者读取/保存 `CareState` 到 MemoryStore。
- 理赔模块是方案建议原型，不替代财务、风控或合规审批。
- FastAPI/SSE、静态工作台和结构化观测代码已加入，但 API 依赖、Redis、真实模型和生产业务 API 仍需在目标环境单独做集成验证。

## 项目验收与协作方式

当前项目的生产深度盘点见 [项目最终验收标准_压力面试矩阵.md](项目最终验收标准_压力面试矩阵.md)，后续开发和阶段提交遵循 [VibeCoding协作契约.md](VibeCoding协作契约.md)。T18-T22 已建立本地多 Skill 计划/DAG、结果校验、只读有限恢复和评测基线；后续优先改进多意图/冲突决策并获取真实预测对照，再补确认闭环、持久化幂等和生产适配。

