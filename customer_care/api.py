"""客服 Agent 的 FastAPI 与 SSE 适配层。"""

import json
from collections.abc import AsyncIterator
from typing import Any

from common.llm import create_llm_client
from customer_care.service import CustomerCareService


def create_app(service: CustomerCareService | None = None) -> Any:
    """创建 API 应用；Web 依赖未安装时给出明确安装提示。"""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
        from starlette.responses import StreamingResponse
    except ImportError as exc:
        raise RuntimeError("请执行 uv sync --extra web 安装 API 依赖") from exc

    app = FastAPI(title="Customer Service Agent", version="0.1.0")
    service = service or CustomerCareService(create_llm_client())

    class ChatRequest(BaseModel):
        user_id: str = Field(min_length=1, max_length=128)
        content: str = Field(min_length=1, max_length=4000)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "customer-care"}

    @app.post("/v1/chat")
    async def chat(request: ChatRequest) -> dict[str, Any]:
        try:
            state, trace = await service.chat(request.user_id, request.content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"answer": state.answer, "trace": trace}

    @app.get("/v1/chat/{user_id}/stream")
    async def stream(user_id: str, content: str) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            try:
                state, trace = await service.chat(user_id, content)
                for event in ({"type": "trace", "data": trace}, {"type": "answer", "data": state.answer}):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'data': str(exc)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app
