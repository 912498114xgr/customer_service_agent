# 客服切片 T22：多 Skill 评测基线

## 本阶段目标与边界

此前 T9 只测单 Skill 的候选召回，T11 的轨迹评测则用脚本预置模型响应，验证的是流程契约，不是规划模型对自然语言的真实理解。T22 为多任务入口、计划、依赖、冲突和错选建立固定回归集与统一评分接口，并测量当前图中连接词检测器的实际表现。

样本为人工编写的 25 条回归案例，不是线上流量，也不具备统计代表性。由于没有在本阶段调用真实模型，T22 不报告 LLM 的计划准确率；评分器通过测试也不代表模型已通过。

## 文件变更

| 文件 | 变更 | 原因 |
|---|---|---|
| `customer_care/orchestration/detection.py` | 抽出主图正在使用的连接词检测函数 | 让评测运行与生产路径中的启发式完全一致 |
| `customer_care/graph.py` | 改为调用抽出的检测函数 | 保持图中行为与基线评测同源，避免评测复制一份逻辑 |
| `customer_care/evaluation/orchestration_dataset.json` | 新增 v1.0.0 的 25 条多语言标注样本及期望工具/参数/依赖/澄清/冲突标签 | 覆盖多任务、隐式请求、冲突、缺参、否定消歧和单任务连接词误报 |
| `customer_care/evaluation/orchestration.py` | 增加数据集校验、入口分类指标、多维分组和计划预测评分器 | 可统一比较不同检测/规划版本，并输出 bad cases |
| `customer_care/evaluation/__main__.py` | 将多 Skill 基线接入 `python -m customer_care.evaluation` | 从项目主评测命令即可看到单 Skill 与多 Skill 基线 |
| `tests/test_orchestration_evaluation.py` | 验证检测器坏例、完美计划评分、冲突执行、参数错配、副作用越权和非法预测输入 | 防止评测器本身算错或悄悄接受损坏数据 |
| `README.md`、`项目最终验收标准_压力面试矩阵.md`、`VibeCoding协作契约.md` | 更新阶段、证据和剩余缺口 | 防止将评测框架的完成误写成模型能力完成 |

## 当前连接词检测器基线

在 25 条固定样本上：

| 指标 | 结果 |
|---|---:|
| Accuracy | 52.0% |
| Precision | 66.7% |
| Recall | 50.0% |
| F1 | 57.1% |

坏例能定位具体问题：

- 误报 4 条：句子里有 `and`/`plus`，但实际只有一个客服目标。
- 漏报 8 条：包括隐式多任务、中文“顺便”、西语 `y`/`también`，以及没有连接词的互相矛盾指令。
- 西语 3 条样本上该检测器没有命中；隐式多任务 4 条和冲突 2 条均没有命中。

这证明当前入口规则不能承担多 Skill 语义路由。样本量有限，结果用于暴露设计缺口和做版本回归，不外推到生产准确率。

## 运行与评分真实计划

运行现有检测器基线和项目其它离线评测：

```powershell
uv run python -m customer_care.evaluation
```

对真实模型导出的预测文件评分，不触发模型调用：

```powershell
uv run python -m customer_care.evaluation --predictions .\artifacts\orchestration-predictions.json
```

预测文件每项格式：

```json
[
  {
    "id": "multi-order-status-receipt",
    "tool_names": ["get_order_status", "get_order_receipt"],
    "task_arguments": [{"order_id": "A"}, {"order_id": "A"}],
    "dependency_edges": [],
    "needs_clarification": false,
    "unconfirmed_side_effect_tools": []
  }
]
```

评分器输出工具 Precision/Recall/F1、工具+参数任务 Precision/Recall/F1、参数完全匹配率、依赖边 F1、整计划完全匹配率、澄清准确率、冲突召回，以及冲突下仍执行工具/未确认副作用计数。缺失预测按空计划计为失败；重复或未知 case ID 会直接报错。

## 验证结果

```text
uv --cache-dir .uv-cache run --no-sync pytest -q tests/test_orchestration_evaluation.py
5 passed

uv --cache-dir .uv-cache run --no-sync pytest -q
65 passed

uv --cache-dir .uv-cache run --no-sync python -m customer_care.evaluation
T9 单工具路由：40 条，Top-1 70.0%，Recall@3 92.5%，Recall@10 100.0%
T22 多 Skill 入口：25 条，Accuracy 52.0%，Precision 66.7%，Recall 50.0%，F1 57.1%
T11 轨迹契约：8 条，固定脚本样例通过率 100.0%（不是 LLM 规划准确率）
```

## 自审结论

当前可回答“怎样知道路由变好了”：固定同一数据集，对比版本指标和逐条 bad case，并让计划字段使用完全相同的评分协议。当前还不能回答“真实 LLM 准确率是多少”，因为没有真实模型预测结果；T9 的 70% Top-1 是单工具候选召回，也不能替代多 Skill 规划准确率。

下一阶段不应只添加更多规则。应针对当前漏报/误报选择路由策略，明确模型调用成本和澄清/冲突优先级，再以本基线做对照实验；同时必须保留真实模型结果和使用的模型/提示版本，Mock 测试只能验证 harness 与安全契约。
