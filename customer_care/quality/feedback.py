"""质检与人工协作反馈闭环。"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from customer_care.quality.scoring import QualityResult
from customer_care.state import CareState


@dataclass(frozen=True)
class HumanFeedback:
    """人工坐席对一轮 Agent 结果的结构化反馈。"""

    user_id: str
    accepted: bool
    corrected_answer: str | None = None
    reason: str = ""
    created_at: str = ""

    @classmethod
    def create(cls, user_id: str, accepted: bool, *, corrected_answer: str | None = None, reason: str = "") -> "HumanFeedback":
        return cls(user_id, accepted, corrected_answer, reason, datetime.now(timezone.utc).isoformat())


def handoff_recommendation(state: CareState, quality: QualityResult | None = None) -> dict[str, Any]:
    """根据状态和质检结果生成坐席工作项，不执行外部转接。"""
    reasons: list[str] = []
    if state.route == "human":
        reasons.append("用户情绪或意图置信度触发人工分支")
    if state.route == "clarify":
        reasons.append("上下文不足，需要人工确认问题范围")
    if quality and not quality.passed:
        reasons.append("质检未通过")
    return {
        "requires_human": bool(reasons),
        "priority": "high" if state.emotion in {"angry", "upset"} else "normal",
        "reasons": reasons,
        "summary": state.handoff_summary or "",
    }


def feedback_to_eval_record(state: CareState, feedback: HumanFeedback) -> dict[str, Any]:
    """将人工修正转换成后续评测集可使用的记录。"""
    user_messages = [message for message in state.messages if message.get("role") == "user"]
    query = str(user_messages[-1].get("content", "")) if user_messages else ""
    return {
        "id": f"feedback-{feedback.created_at}",
        "query": query,
        "intent": state.intent or "other",
        "route": state.route,
        "expected_tool": state.selected_tool,
        "human_accepted": feedback.accepted,
        "corrected_answer": feedback.corrected_answer,
        "reason": feedback.reason,
    }


def append_feedback(path: Path, state: CareState, feedback: HumanFeedback) -> None:
    """以 JSONL 追加反馈，便于后续导入标注和评测流程。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(feedback_to_eval_record(state, feedback), ensure_ascii=False) + "\n")
