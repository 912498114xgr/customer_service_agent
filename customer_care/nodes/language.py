"""客服输入语言识别节点。"""

import re

from customer_care.state import CareState

_SPANISH_MARKERS = re.compile(r"\b(hola|gracias|dónde|cuándo|devolución|quiero|está)\b", re.I)
_PORTUGUESE_MARKERS = re.compile(r"\b(olá|obrigado|você|onde|quando|devolução|estou|meu|minha)\b", re.I)


def detect_language(state: CareState) -> dict[str, str]:
    """根据常见词语与字符特征识别西语、葡语或英语。"""
    query = latest_user_message(state)
    if _PORTUGUESE_MARKERS.search(query) or re.search(r"[ãõç]", query, re.I):
        language = "pt"
    elif _SPANISH_MARKERS.search(query) or re.search(r"[¿¡ñ]", query, re.I):
        language = "es"
    else:
        language = "en"
    return {"language": language}


def latest_user_message(state: CareState) -> str:
    """取最近一条用户消息；缺失时返回空字符串。"""
    for message in reversed(state.messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


