"""客服 Agent 端到端轨迹契约评测。"""

import asyncio
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.llm import ChatMessage, LLMClient
from customer_care.graph import build_customer_care_graph
from customer_care.state import CareState


@dataclass(frozen=True)
class TrajectoryCase:
    """一条轨迹样本的人工期望。"""

    case_id: str
    query: str
    intent: str
    route: str
    expected_tool: str | None
    entities: dict[str, Any]
    emotion: str


class ScriptedLLM(LLMClient):
    """按预置响应驱动图，验证图契约而不访问真实模型。"""

    def __init__(self, responses: list[str]) -> None:
        self._responses = deque(responses)
        self.calls: list[list[ChatMessage]] = []

    async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.0) -> str:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("轨迹评测消耗了超出预期的模型调用")
        return self._responses.popleft()


def load_trajectory_dataset(path: Path | None = None) -> list[TrajectoryCase]:
    """加载并校验端到端轨迹标签。"""
    path = path or Path(__file__).with_name("trajectory_dataset.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        TrajectoryCase(
            case_id=item["id"],
            query=item["query"],
            intent=item["intent"],
            route=item["route"],
            expected_tool=item["expected_tool"],
            entities=item.get("entities", {}),
            emotion=item.get("emotion", "neutral"),
        )
        for item in raw
    ]
    ids = [case.case_id for case in cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError("轨迹评测集不能为空，且 id 必须唯一")
    return cases


def evaluate_trajectory(cases: list[TrajectoryCase] | None = None) -> dict[str, Any]:
    """离线回放图并统计意图、路由和工具契约命中率。"""
    cases = cases or load_trajectory_dataset()
    results = [_run_case(case) for case in cases]
    total = len(results)
    return {
        "total": total,
        "intent_accuracy": sum(item["intent_ok"] for item in results) / total,
        "route_accuracy": sum(item["route_ok"] for item in results) / total,
        "tool_accuracy": sum(item["tool_ok"] for item in results) / total,
        "all_contracts_accuracy": sum(item["all_ok"] for item in results) / total,
        "bad_cases": [item for item in results if not item["all_ok"]],
        "cases": results,
    }


def format_trajectory_report(report: dict[str, Any]) -> str:
    """格式化轨迹评测摘要。"""
    return "\n".join(
        [
            "客服 Agent 轨迹契约评测",
            f"样本数：{report['total']}",
            f"意图准确率：{report['intent_accuracy']:.1%}",
            f"路由准确率：{report['route_accuracy']:.1%}",
            f"工具选择准确率：{report['tool_accuracy']:.1%}",
            f"全链路契约通过率：{report['all_contracts_accuracy']:.1%}",
        ]
    )


def _run_case(case: TrajectoryCase) -> dict[str, Any]:
    llm = ScriptedLLM(_responses_for(case))
    graph = build_customer_care_graph(llm)
    state = CareState(user_id="eval-user", messages=[{"role": "user", "content": case.query}])
    result = CareState.model_validate(asyncio.run(graph.ainvoke(state.model_dump())))
    actual_tool = result.selected_tool if result.route == "self_service" else None
    intent_ok = result.intent == case.intent
    route_ok = result.route == case.route
    tool_ok = actual_tool == case.expected_tool
    return {
        "id": case.case_id,
        "query": case.query,
        "expected": {"intent": case.intent, "route": case.route, "tool": case.expected_tool},
        "actual": {"intent": result.intent, "route": result.route, "tool": actual_tool},
        "intent_ok": intent_ok,
        "route_ok": route_ok,
        "tool_ok": tool_ok,
        "all_ok": intent_ok and route_ok and tool_ok,
        "model_calls": len(llm.calls),
    }


def _responses_for(case: TrajectoryCase) -> list[str]:
    responses = [
        json.dumps(
            {
                "intent": case.intent,
                "entities": case.entities,
                "emotion": case.emotion,
                "confidence": 0.95 if case.intent != "other" else 0.2,
                "goals": [{"intent": case.intent, "objective": case.query}],
                "conflict": False,
                "needs_clarification": case.intent == "other",
            }
        )
    ]
    if "that" in case.query.lower():
        responses.append(
            json.dumps(
                {
                    "rewritten_query": case.query,
                    "ambiguous": True,
                    "explanation": "没有唯一的指代对象。",
                }
            )
        )
    if case.route == "self_service":
        responses.append(json.dumps({"tool_name": case.expected_tool, "arguments": case.entities}))
    responses.append("这是轨迹评测的固定回复。")
    return responses
