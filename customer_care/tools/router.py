"""两级客服工具路由：词袋粗筛与候选内模型细选。"""

import logging
import re
from collections import OrderedDict
from collections.abc import Sequence

from pydantic import BaseModel, Field

from common.llm import LLMClient
from common.structured_output import parse_model_json
from customer_care.nodes.language import latest_user_message
from customer_care.state import CareState
from customer_care.tools.registry import ToolCatalog, ToolDefinition

_LOGGER = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9áéíóúüñçãõàèìòùêô]+", re.I)
_INTENT_CATEGORY = {
    "refund": "refunds",
    "track_order": "delivery",
    "order": "orders",
    "coupon": "coupons",
    "account": "account",
}


class ToolChoice(BaseModel):
    tool_name: str
    arguments: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class ToolRouter:
    def __init__(
        self,
        llm: LLMClient,
        catalog: ToolCatalog | None = None,
        *,
        max_candidates: int = 10,
        cache_size: int = 256,
    ) -> None:
        if max_candidates < 1 or max_candidates > 10:
            raise ValueError("max_candidates 必须在 1 到 10 之间")
        self._llm = llm
        self._catalog = catalog or ToolCatalog.load_default()
        self._max_candidates = max_candidates
        self._cache_size = cache_size
        self._candidate_cache: OrderedDict[str, tuple[str, ...]] = OrderedDict()

    async def select(self, state: CareState) -> dict[str, object]:
        query = state.rewritten_query or latest_user_message(state)
        cache_key = f"{state.intent or 'other'}:{' '.join(query.lower().split())}"
        names = self._candidate_cache.get(cache_key)
        if names is None:
            candidates = self.coarse_candidates(query, state.intent)
            names = tuple(tool.name for tool in candidates)
            self._candidate_cache[cache_key] = names
            self._candidate_cache.move_to_end(cache_key)
            while len(self._candidate_cache) > self._cache_size:
                self._candidate_cache.popitem(last=False)
        else:
            self._candidate_cache.move_to_end(cache_key)

        candidate_by_name = {tool.name: tool for tool in self._catalog.tools if tool.name in names}
        candidates = [candidate_by_name[name] for name in names]
        choice = await self._fine_select(query, state, candidates)
        selected = candidate_by_name.get(choice.tool_name, candidates[0])
        arguments = self._validated_arguments(selected, choice.arguments, state)
        return {
            "selected_tool": selected.name,
            "tool_args": arguments,
        }

    def coarse_candidates(self, query: str, intent: str | None) -> list[ToolDefinition]:
        """返回粗筛候选，供评测和观测使用；不会调用模型或执行工具。"""
        query_tokens = set(_TOKEN.findall(query.lower()))
        preferred_category = _INTENT_CATEGORY.get(intent or "")
        ranked: list[tuple[float, int, ToolDefinition]] = []
        for index, tool in enumerate(self._catalog.tools):
            keyword_tokens = set(_TOKEN.findall(" ".join(tool.trigger_keywords).lower()))
            document_tokens = set(
                _TOKEN.findall(
                    " ".join(
                        [tool.description, *tool.trigger_keywords, *tool.pos_examples]
                    ).lower()
                )
            )
            exact_phrase_hits = sum(
                2.0 for keyword in tool.trigger_keywords if keyword.lower() in query.lower()
            )
            overlap = len(query_tokens & document_tokens)
            union = len(query_tokens | document_tokens) or 1
            bag_similarity = len(query_tokens & document_tokens) / union
            keyword_overlap = len(query_tokens & keyword_tokens)
            category_bonus = 1.0 if tool.category == preferred_category else 0.0
            score = exact_phrase_hits + keyword_overlap + bag_similarity + category_bonus
            if score > 0:
                ranked.append((score, index, tool))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        if not ranked:
            ranked = [
                (1.0 if tool.category == preferred_category else 0.0, index, tool)
                for index, tool in enumerate(self._catalog.tools)
            ]
            ranked.sort(key=lambda item: (-item[0], item[1]))
        return [tool for _, _, tool in ranked[: self._max_candidates]]

    def _coarse_filter(self, query: str, intent: str | None) -> list[ToolDefinition]:
        """保留旧的内部方法名，避免已有调用方因为观测接口升级而失效。"""
        return self.coarse_candidates(query, intent)

    async def _fine_select(
        self,
        query: str,
        state: CareState,
        candidates: Sequence[ToolDefinition],
    ) -> ToolChoice:
        descriptions = [
            {
                "name": tool.name,
                "category": tool.category,
                "description": tool.description,
                "allowed_arguments": tool.argument_names,
                "positive_examples": tool.pos_examples,
                "negative_examples": tool.neg_examples,
            }
            for tool in candidates
        ]
        prompt = (
            "从给定候选中选择最匹配的一个工具，只返回 JSON。\n"
            '{"tool_name":"候选名称","arguments":{}}\n'
            "只能选择候选名称，只能填写 allowed_arguments 中的参数；不确定时选最保守的只读查询。\n"
            f"用户问题：{query}\n意图：{state.intent}\n已知实体：{state.entities}\n候选：{descriptions}"
        )
        try:
            return parse_model_json(
                await self._llm.complete([{"role": "user", "content": prompt}]),
                ToolChoice,
            )
        except Exception:
            _LOGGER.exception("工具细选调用或输出校验失败，使用粗筛第一候选")
            return ToolChoice(tool_name=candidates[0].name)

    @staticmethod
    def _validated_arguments(
        tool: ToolDefinition,
        proposed: dict[str, str | int | float | bool | None],
        state: CareState,
    ) -> dict[str, str | int | float | bool | None]:
        arguments = {key: value for key, value in proposed.items() if key in tool.argument_names}
        for key in tool.argument_names:
            if key not in arguments and key in state.entities:
                value = state.entities[key]
                if isinstance(value, (str, int, float, bool)) or value is None:
                    arguments[key] = value
        return arguments

