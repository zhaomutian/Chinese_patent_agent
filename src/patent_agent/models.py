import json
import math
import time

import httpx
from langchain_openai import ChatOpenAI

from .settings import Settings


class ModelError(RuntimeError):
    pass


def safe_error(exc):
    # Never log request URLs, headers, provider bodies or exception reprs.
    code = getattr(exc, "status_code", None)
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
    return f"{type(exc).__name__}" + (f" (HTTP {code})" if code else "")


def chat_client(s: Settings, timeout=None):
    return ChatOpenAI(
        model=s.chat_model,
        base_url=s.chat_base_url.get_secret_value(),
        api_key=s.chat_api_key.get_secret_value(),
        use_responses_api=True,
        reasoning={"effort": s.chat_reasoning_effort},
        temperature=s.chat_temperature if s.chat_send_temperature else None,
        max_tokens=s.chat_max_output_tokens,
        timeout=timeout or s.model_timeout_seconds,
        max_retries=s.model_max_retries,
        store=False,
    )


def message_text(message):
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") in ("text", "output_text")
    )


def parse_json(message):
    meta = message.response_metadata
    if meta.get("status") == "incomplete" or meta.get("finish_reason") == "length":
        raise ModelError("模型输出被截断")
    value = message_text(message).strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(value)


class Embeddings:
    def __init__(self, settings: Settings):
        self.s = settings
        self.requests = 0
        self.tokens = 0

    def encode(self, texts: list[str], timeout=None) -> list[list[float]]:
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("向量输入必须为非空文本列表")
        payload = {
            "model": self.s.embedding_model,
            "input": texts,
            "encoding_format": self.s.embedding_encoding_format,
        }
        if self.s.embedding_send_dimensions:
            payload["dimensions"] = self.s.embedding_dimensions
        # This is the only embedding retry layer. Never retry 4xx except 429.
        for attempt in range(self.s.model_max_retries + 1):
            try:
                self.requests += 1
                r = httpx.post(
                    self.s.embedding_base_url.rstrip("/") + "/embeddings",
                    headers={"Authorization": "Bearer " + self.s.embedding_api_key.get_secret_value()},
                    json=payload,
                    timeout=timeout or self.s.model_timeout_seconds,
                )
                r.raise_for_status()
                body = r.json()
                break
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                retryable = not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code in (
                    429,
                    500,
                    502,
                    503,
                    504,
                )
                if not retryable or attempt == self.s.model_max_retries:
                    raise ModelError(safe_error(exc)) from None
                time.sleep(min(2**attempt, 4))
        rows = sorted(body.get("data", []), key=lambda x: x["index"])
        if [row["index"] for row in rows] != list(range(len(texts))):
            raise ModelError("向量响应数量或顺序索引异常")
        vectors = [row["embedding"] for row in rows]
        for vector in vectors:
            if len(vector) != self.s.embedding_dimensions or not all(
                isinstance(x, (int, float)) and math.isfinite(x) for x in vector
            ):
                raise ModelError("向量维度或数值异常")
            if not any(vector):
                raise ModelError("零向量不可入库")
        self.tokens += body.get("usage", {}).get("total_tokens", 0)
        return vectors
