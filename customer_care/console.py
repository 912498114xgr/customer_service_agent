"""客服 Agent 交互式终端入口。"""

import argparse
import asyncio
import json
import os

from common.config import Settings
from common.llm import create_llm_client
from common.mock_llm import MockLLM
from customer_care.memory.sqlite_memory import SQLiteMemoryStore
from customer_care.service import CustomerCareService


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="与客服 Agent 进行交互式对话")
    parser.add_argument("--user-id", default="demo-user", help="会话用户标识")
    return parser.parse_args()


async def _run(user_id: str) -> None:
    settings = Settings()
    try:
        llm = create_llm_client(settings)
    except ValueError as exc:
        print(f"模型配置错误：{exc}")
        print("请检查 .env 中的 USE_MOCK、LLM_API_KEY、LLM_BASE_URL 和 LLM_MODEL。")
        return
    memory = SQLiteMemoryStore(settings.sqlite_memory_path) if settings.sqlite_memory_path else None
    service = CustomerCareService(llm, memory=memory)

    print("客服 Agent 已启动。输入 /help 查看命令，输入 /quit 退出。")
    if isinstance(llm, MockLLM):
        print("当前 USE_MOCK=true：这是离线演示模式，真实自由问答请配置 LLM_API_KEY 并关闭 Mock。")
    print(f"当前用户：{user_id}\n")

    while True:
        try:
            content = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return
        if not content:
            continue
        if content in {"/quit", "/exit"}:
            print("已退出。")
            return
        if content == "/help":
            print("/trace 查看上一轮决策链路，/reset 清空当前会话，/quit 退出。\n")
            continue
        if content == "/reset":
            await service.reset(user_id)
            print("当前会话已重置。\n")
            continue
        if content == "/trace":
            print("请在一轮对话后使用 /trace；当前版本会在每次回复后自动展示。\n")
            continue

        try:
            state, trace = await service.chat(user_id, content)
        except Exception as exc:
            print(f"Agent 处理失败：{exc}\n")
            continue
        print(f"Agent> {state.answer or '(没有生成回复)'}")
        print("[本轮决策链路]")
        print(json.dumps(trace, ensure_ascii=False, indent=2, default=str))
        print()


def main() -> None:
    """启动交互式客服终端。"""
    args = _parse_args()
    asyncio.run(_run(args.user_id))


if __name__ == "__main__":
    main()

