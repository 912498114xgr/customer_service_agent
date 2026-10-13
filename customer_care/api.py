"""客服 Agent 的 FastAPI 与 SSE 适配层。"""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from common.llm import create_llm_client
from common.config import Settings
from customer_care.demo import create_scenario_llm, demo_scenarios
from customer_care.service import CustomerCareService


def create_app(service: CustomerCareService | None = None) -> Any:
    """创建 API 应用；Web 依赖未安装时给出明确安装提示。"""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
        from starlette.responses import FileResponse
        from starlette.responses import StreamingResponse
    except ImportError as exc:
        raise RuntimeError("请执行 uv sync --extra web 安装 API 依赖") from exc

    app = FastAPI(title="Customer Service Agent", version="0.1.0")
    settings = Settings()
    service = service or CustomerCareService(create_llm_client(settings))

    class ChatRequest(BaseModel):
        user_id: str = Field(min_length=1, max_length=128)
        content: str = Field(min_length=1, max_length=4000)

    @app.get("/", include_in_schema=False)
    async def workbench() -> FileResponse:
        return FileResponse(Path(__file__).with_name("web") / "index.html")

    @app.get("/health")
    async def health() -> dict[str, str | bool]:
        return {
            "status": "ok",
            "service": "customer-care",
            "mode": "mock" if settings.use_mock else "live",
            "live_configured": bool(settings.llm_api_key),
        }

    @app.get("/v1/demo/scenarios")
    async def scenarios() -> dict[str, Any]:
        return {"mode": "scripted_mock", "scenarios": demo_scenarios()}

    @app.post("/v1/demo/{scenario_id}")
    async def run_demo(scenario_id: str) -> dict[str, Any]:
        from fastapi import HTTPException

        scenarios_by_id = {scenario["id"]: scenario for scenario in demo_scenarios()}
        scenario = scenarios_by_id.get(scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="演练场景不存在")
        demo_service = CustomerCareService(create_scenario_llm(scenario_id))
        state, trace = await demo_service.chat(f"scripted-demo-{scenario_id}", scenario["query"])
        return {
            "answer": state.answer,
            "trace": trace,
            "demo": {"mode": "scripted_mock", **scenario},
        }

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
