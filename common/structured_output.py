"""从模型响应中读取并校验结构化 JSON。"""

import json
from typing import TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def parse_model_json(raw: str, model_type: type[ModelT]) -> ModelT:
    """解析纯 JSON 或 Markdown 代码块中的第一个 JSON 对象。"""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", maxsplit=1)[-1]
        if text.endswith("```"):
            text = text[:-3].strip()

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character == "{":
            try:
                payload, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            return model_type.model_validate(payload)
    raise ValueError("模型响应中没有 JSON 对象")

