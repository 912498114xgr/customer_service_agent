"""对客服回复执行确定性的合规红线检查。"""

import re
from typing import Any

from pydantic import BaseModel


class RuleHit(BaseModel):
    rule_id: str
    description: str
    message_index: int | None = None


_INSULTS = re.compile(
    r"\b(stupid|idiot|shut up|moron|estúpido|idiota|cállate|burro|burra|"
    r"imbecil|imbécil|cala a boca)\b",
    re.I,
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_PHONE_WITH_COUNTRY = re.compile(r"(?<!\w)\+\d[\d -]{7,}\d(?!\w)")
_PROMISE = re.compile(
    r"\b(we will refund|i will refund|guarantee a refund|le devolveremos|"
    r"le reembolsaremos|vamos reembolsar|garantimos o reembolso)\b",
    re.I,
)
_COMPLETION = re.compile(
    r"\b(refund (?:has been|was) processed|refund completed|reembolso realizado|"
    r"reembolso processado|reembolso concluído)\b",
    re.I,
)


def check_hard_rules(
    messages: list[dict[str, Any]],
    *,
    max_response_delay_seconds: float = 60.0,
) -> list[RuleHit]:
    """检查辱骂、隐私泄露、未兑现承诺和超时未回等硬性规则。"""
    hits: list[RuleHit] = []
    assistant_messages = [
        (index, message)
        for index, message in enumerate(messages)
        if message.get("role") in {"assistant", "agent"}
    ]
    assistant_text = "\n".join(str(message.get("content", "")) for _, message in assistant_messages)

    for index, message in assistant_messages:
        content = str(message.get("content", ""))
        if _INSULTS.search(content):
            hits.append(RuleHit(rule_id="insult", description="客服回复包含侮辱性表达。", message_index=index))
        if _EMAIL.search(content) or _CARD.search(content) or _PHONE_WITH_COUNTRY.search(content):
            hits.append(RuleHit(rule_id="privacy_leak", description="客服回复疑似泄露邮箱、卡号或电话。", message_index=index))

    if _PROMISE.search(assistant_text) and not _COMPLETION.search(assistant_text):
        hits.append(RuleHit(rule_id="unfulfilled_promise", description="客服作出退款承诺，但会话中没有完成凭证。"))

    for index, message in enumerate(messages):
        delay = message.get("response_delay_seconds")
        if isinstance(delay, (int, float)) and delay > max_response_delay_seconds:
            hits.append(RuleHit(rule_id="response_timeout", description="客服响应超过规定时限。", message_index=index))

    return hits

