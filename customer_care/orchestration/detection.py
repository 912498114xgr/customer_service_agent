"""多任务入口的轻量候选检测。"""

_MULTI_TASK_MARKERS = (" and ", " also ", " plus ", "同时", "以及", "另外", "如果", " then ")


def is_multi_task_request(query: str) -> bool:
    """连接词基线；只用于选择候选路径，不能代替任务语义拆解。"""
    normalized = query.lower()
    return any(marker in normalized for marker in _MULTI_TASK_MARKERS)
