from customer_care.knowledge import KnowledgeRetriever


def test_knowledge_retriever_returns_citations_for_refund_query() -> None:
    retriever = KnowledgeRetriever.load_default()

    hits = retriever.search("When will my refund arrive?", language="en", topic="refund")

    assert hits
    assert hits[0].document_id == "refund-timeline"
    assert hits[0].score > 0
    assert "payment provider" in retriever.context(hits)


def test_knowledge_retriever_handles_chinese_coupon_query() -> None:
    retriever = KnowledgeRetriever.load_default()

    hits = retriever.search("优惠券为什么不能使用", language="zh", topic="coupon")

    assert hits
    assert hits[0].document_id == "coupon-help"


def test_knowledge_retriever_rejects_invalid_top_k() -> None:
    retriever = KnowledgeRetriever.load_default()

    try:
        retriever.search("refund", top_k=0)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("expected invalid top_k to fail")
