# 客服切片：T14 API 与 SSE

新增 `customer_care/api.py`，复用 `CustomerCareService` 提供 `/health`、`POST /v1/chat` 和 SSE 流式端点。Web 依赖放在 `web` 可选组，不影响离线核心测试。

安装并启动：

```powershell
uv sync --extra web
uv run uvicorn customer_care.api:create_app --factory --reload
```

当前环境因网络权限未完成 FastAPI 下载；核心测试保持通过，API 依赖安装后即可启动。
