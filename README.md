# Agent Rebuild

用 Python + LangGraph 重实现两个本地可验证的 Agent 垂直切片。当前客服切片推进到 T26；诊断切片尚未开始。T18-T26 加入本地多 Skill 计划/DAG 执行、主图接入、结果契约、只读有限恢复、多目标意图契约、固定评测基线、逐步解释的本地学习工作台、请求级 LLM 观测和副作用确认闭环；尚无真实模型预测指标，不能视作生产级路由。

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
uv sync --default-index https://pypi.org/simple --extra web
uv run uvicorn customer_care.api:create_app --factory --reload
```

打开 `http://127.0.0.1:8000` 可用学习工作台体验客服主图。页面的固定剧本会真实运行本地 LangGraph 和工具注册表，但模型响应是预置的，用来观察多 Skill、冲突拦截、结构化输出失败和副作用确认/拒绝分支；它不是自由问答，也不代表模型准确率。学习路线和面试自测问题见 [T24 学习工作台说明](客服切片_T24_学习工作台与主链路演练.md)；请求级调用数、节点耗时和 token 粗估见 [T25 请求级模型观测](客服切片_T25_请求级模型观测.md)，确认令牌状态机及限制见 [T26 副作用操作确认与恢复](客服切片_T26_副作用确认与恢复.md)。要自由提问，需要在 `.env` 配置兼容模型，并设置 `USE_MOCK=false`；真实请求可能产生费用。本地没有 `.env` 时默认只显示固定剧本，页面会明确提示真实模型未配置。

API 端点为 `/health`、`POST /v1/chat`、`POST /v1/chat/{user_id}/confirm|cancel`、`GET /v1/chat/{user_id}/stream` 和固定演练 `/v1/demo/*`（演练确认/拒绝端点为 `/v1/demo/{scenario_id}/confirm|cancel`）；工作台文件在 `customer_care/web/index.html`。Web 依赖属于 `web` extra，不安装时核心离线主图演练仍可直接测试。

## 路由评测基线

T9 新增了 40 条人工标注的工具路由样本，覆盖 30 个工具、三种语言和近义混淆。运行：

```powershell
uv run python -m customer_care.evaluation
```

当前离线基线为 `Top-1=70.0%`、`Recall@3=92.5%`、`Recall@10=100.0%`。Recall@3 用来衡量第二级模型收到的候选是否包含正确工具，不能当作端到端回答准确率。逐条 bad case 和类别分项结果见 [客服切片_T9_评测基线.md](客服切片_T9_评测基线.md)。

T22 增加了 25 条多 Skill 固定回归样本和计划评分器。T23 已移除主图中的连接词入口判断，在现有意图模型调用中输出多个目标、冲突及需澄清状态；T22 的启发式结果（Accuracy=52.0%、F1=57.1%）保留为历史对照，不代表 T23 或真实模型效果。通过 `--intent-predictions` 可评分外部导出的意图预测，但本仓库尚无真实模型预测结果。完整范围见 [客服切片_T23_多目标意图与冲突路由.md](客服切片_T23_多目标意图与冲突路由.md)。

当前提供真实模型固定集评测运行器，复用意图识别、单 Skill 路由及多 Skill Planner，只生成并评分决策，不执行任何工具。真实模型调用默认关闭；只有本机 `.env` 设置 `USE_MOCK=false` 并提供模型凭据，再显式给出授权、样本上限和输出路径才会运行：

```powershell
uv run python -m customer_care.evaluation.live --allow-model-calls --max-cases 3 --output .\artifacts\model-evaluation.json
```

每条样本最多两个逻辑模型调用，SDK 按 `LLM_MAX_RETRIES` 可能进行额外传输重试。报告记录模型名、数据集与提示源哈希、逻辑调用数、字符量/token 粗估、分项指标和 bad case，不记录密钥或提示正文。小于全量的 `--max-cases` 结果只代表数据集前缀子集，不代表完整指标；固定人工样本也不代表生产流量。详见 [客服切片_T27_真实模型评测运行器.md](客服切片_T27_真实模型评测运行器.md)。

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

当前项目的生产深度盘点见 [项目最终验收标准_压力面试矩阵.md](项目最终验收标准_压力面试矩阵.md)，后续开发和阶段提交遵循 [VibeCoding协作契约.md](VibeCoding协作契约.md)。T18-T27 已建立本地多 Skill 计划/DAG、多目标路由契约、冲突澄清、结果校验、只读有限恢复、固定集模型评测运行器、主图演练、请求级模型调用观测和本地副作用确认流程；T27 尚未获得真实模型运行结果。后续仍需模型预测对照、持久化原子状态迁移、跨进程幂等和生产适配。

