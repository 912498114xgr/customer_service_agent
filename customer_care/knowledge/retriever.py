"""基于 TF-IDF 词项和字符片段的可解释混合检索。"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TOKEN = re.compile(r"[\w]+", re.UNICODE)


@dataclass(frozen=True)
class KnowledgeDocument:
    document_id: str
    title: str
    language: str
    topics: tuple[str, ...]
    content: str


@dataclass(frozen=True)
class KnowledgeHit:
    document_id: str
    title: str
    language: str
    snippet: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        """转为可安全放入状态和 API 响应的来源信息。"""
        return {
            "id": self.document_id,
            "title": self.title,
            "language": self.language,
            "snippet": self.snippet,
            "score": round(self.score, 4),
        }


class KnowledgeRetriever:
    """本地知识库检索器；不访问网络，结果带来源和分数。"""

    def __init__(self, documents: list[KnowledgeDocument], *, min_score: float = 0.08) -> None:
        if not documents:
            raise ValueError("知识库不能为空")
        self._documents = tuple(documents)
        self._min_score = min_score
        self._tokens = {doc.document_id: _tokens(f"{doc.title} {doc.content} {' '.join(doc.topics)}") for doc in documents}
        self._idf = self._build_idf()

    @classmethod
    def load_default(cls) -> "KnowledgeRetriever":
        path = Path(__file__).with_name("documents.json")
        raw = json.loads(path.read_text(encoding="utf-8"))
        documents = [
            KnowledgeDocument(
                document_id=item["id"],
                title=item["title"],
                language=item["language"],
                topics=tuple(item.get("topics", [])),
                content=item["content"],
            )
            for item in raw
        ]
        return cls(documents)

    def search(self, query: str, *, language: str | None = None, topic: str | None = None, top_k: int = 3) -> list[KnowledgeHit]:
        """按词项权重、字符片段和主题匹配排序，并过滤低分噪声。"""
        if top_k < 1:
            raise ValueError("top_k 必须大于 0")
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        query_text = query.lower()
        scored: list[tuple[float, KnowledgeDocument]] = []
        for doc in self._documents:
            if language and doc.language not in {language, "zh"}:
                continue
            token_score = self._token_score(query_tokens, self._tokens[doc.document_id])
            phrase_score = _character_overlap(query_text, doc.content.lower())
            topic_bonus = 0.15 if topic and topic in doc.topics else 0.0
            score = 0.75 * token_score + 0.2 * phrase_score + topic_bonus
            if score >= self._min_score:
                scored.append((score, doc))
        scored.sort(key=lambda item: (-item[0], item[1].document_id))
        return [
            KnowledgeHit(
                document_id=doc.document_id,
                title=doc.title,
                language=doc.language,
                snippet=_snippet(doc.content),
                score=score,
            )
            for score, doc in scored[:top_k]
        ]

    def context(self, hits: list[KnowledgeHit], *, max_chars: int = 2400) -> str:
        """拼装受长度限制的证据上下文，强制回答只使用检索事实。"""
        chunks: list[str] = []
        size = 0
        for hit in hits:
            chunk = f"[{hit.document_id}] {hit.title}: {hit.snippet}"
            if size + len(chunk) > max_chars:
                break
            chunks.append(chunk)
            size += len(chunk)
        return "\n".join(chunks)

    def _build_idf(self) -> dict[str, float]:
        document_count = len(self._documents)
        document_frequency: dict[str, int] = {}
        for tokens in self._tokens.values():
            for token in tokens:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        return {token: math.log((document_count + 1) / (count + 1)) + 1 for token, count in document_frequency.items()}

    def _token_score(self, query_tokens: set[str], document_tokens: set[str]) -> float:
        if not query_tokens or not document_tokens:
            return 0.0
        matched = query_tokens & document_tokens
        numerator = sum(self._idf.get(token, 1.0) for token in matched)
        denominator = sum(self._idf.get(token, 1.0) for token in query_tokens)
        return numerator / denominator if denominator else 0.0


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN.findall(text) if len(token) > 1}


def _character_overlap(query: str, content: str) -> float:
    query_fragments = {query[index : index + 3] for index in range(max(0, len(query) - 2))}
    content_fragments = {content[index : index + 3] for index in range(max(0, len(content) - 2))}
    if not query_fragments:
        return 0.0
    return len(query_fragments & content_fragments) / len(query_fragments)


def _snippet(content: str, max_chars: int = 320) -> str:
    return content if len(content) <= max_chars else content[: max_chars - 3] + "..."
