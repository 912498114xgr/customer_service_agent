"""可重复的主图学习演练；脚本响应不用于评估真实模型能力。"""

from collections import deque
from collections.abc import Sequence

from common.llm import ChatMessage, LLMClient


_SCENARIOS = {
    "multi-skill": {
        "title": "一句话包含两个独立目标",
        "query": "Check order A-17 status and show my receipt",
        "description": "观察意图目标如何进入 Planner，再由 DAG 执行两个只读工具。模型输出是固定脚本。",
        "responses": [
            '{"intent":"order","entities":{"order_id":"A-17"},"emotion":"neutral","confidence":0.96,"goals":[{"intent":"order","objective":"查询订单状态"},{"intent":"order","objective":"获取订单收据"}],"conflict":false,"needs_clarification":false}',
            '{"tasks":[{"task_id":"status","intent":"get_order_status","tool_name":"get_order_status","capability":"get_order_status","arguments":{"order_id":"A-17"},"dependencies":[]},{"task_id":"receipt","intent":"get_order_receipt","tool_name":"get_order_receipt","capability":"get_order_receipt","arguments":{"order_id":"A-17"},"dependencies":[]}],"rationale":"两个独立只读目标","confidence":0.96}',
            "Order A-17 is delivered. The local demo receipt lookup also completed.",
        ],
    },
    "conflict": {
        "title": "矛盾指令先澄清",
        "query": "Cancel order A-17, but keep order A-17 active",
        "description": "观察冲突标志如何在 Planner 和工具执行前截断流程。模型输出是固定脚本。",
        "responses": [
            '{"intent":"order","entities":{"order_id":"A-17"},"emotion":"neutral","confidence":0.96,"goals":[{"intent":"order","objective":"取消订单"},{"intent":"order","objective":"保持订单有效"}],"conflict":true,"needs_clarification":true}',
            "I found conflicting instructions. Should I cancel order A-17, or keep it active?",
        ],
    },
    "side-effect": {
        "title": "副作用工具需要确认",
        "query": "Change my notification preference to weekly",
        "description": "观察单 Skill 路径如何检查工具目录的确认要求，并在执行前暂停。实际注册表仍是本地演示数据。",
        "responses": [
            '{"intent":"account","entities":{},"emotion":"neutral","confidence":0.96,"goals":[{"intent":"account","objective":"更新通知频率"}],"conflict":false,"needs_clarification":false}',
            '{"tool_name":"update_contact_preference","arguments":{"preference":"weekly"}}',
            "This action needs your confirmation before it can be carried out.",
        ],
    },
    "multi-confirmation": {
        "title": "多 Skill 计划等待确认后恢复",
        "query": "Request a review for refund R-17 and explain the refund policy",
        "description": "观察含副作用任务的 DAG 在确认前整体暂停，确认后执行原计划。本地工具不会联系真实退款系统。",
        "responses": [
            '{"intent":"refund","entities":{"order_id":"R-17"},"emotion":"neutral","confidence":0.96,"goals":[{"intent":"refund","objective":"申请人工复核退款"},{"intent":"refund","objective":"了解退款政策"}],"conflict":false,"needs_clarification":false}',
            '{"tasks":[{"task_id":"review","intent":"create_refund_review_ticket","tool_name":"create_refund_review_ticket","capability":"create_refund_review_ticket","arguments":{"order_id":"R-17","reason":"customer requested review"},"dependencies":[]},{"task_id":"policy","intent":"get_refund_policy","tool_name":"get_refund_policy","capability":"get_refund_policy","arguments":{},"dependencies":[]}],"rationale":"一个待确认复核请求和一个只读政策查询","confidence":0.96}',
            "This plan contains a review request and a read-only policy lookup. It needs confirmation before execution.",
        ],
    },
    "invalid-intent": {
        "title": "结构化意图输出损坏时安全降级",
        "query": "Where is order A-17?",
        "description": "观察解析失败如何转澄清，而不是猜测意图并执行工具。模型首个响应故意损坏。",
        "responses": [
            "This is probably an order status question.",
            "I could not safely identify the request, so I did not run an order lookup.",
        ],
    },
}


def demo_scenarios() -> list[dict[str, str]]:
    """返回可在本地重复运行的固定剧本，不包含模型预测成绩。"""
    return [
        {key: str(value) for key, value in scenario.items() if key in {"title", "query", "description"}}
        | {"id": scenario_id}
        for scenario_id, scenario in _SCENARIOS.items()
    ]


def create_scenario_llm(scenario_id: str) -> LLMClient:
    """为指定演练创建严格按序返回的离线模型替身。"""
    try:
        responses = _SCENARIOS[scenario_id]["responses"]
    except KeyError as exc:
        raise ValueError(f"未知演练场景：{scenario_id}") from exc
    return _ScenarioLLM(responses)


class _ScenarioLLM(LLMClient):
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = deque(responses)

    async def complete(self, messages: Sequence[ChatMessage], *, temperature: float = 0.0) -> str:
        if not messages:
            raise ValueError("模型消息不能为空")
        if not self._responses:
            raise RuntimeError("演练响应已耗尽；图发生了未预期的模型调用")
        return self._responses.popleft()
