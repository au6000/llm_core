"""
OpenAI プロバイダー実装
"""

from typing import Iterator

from openai import APIError as OpenAIAPIError, RateLimitError as OpenAIRateLimitError

from ..types import ChatResponse, StreamChunk, Usage
from ..exceptions import LLMAPIError, LLMRateLimitError
from .._init_clients import get_openai_client


def _uses_max_completion_tokens(model: str) -> bool:
    """max_completion_tokensを使用するモデルかどうかを判定

    対象: o1, o3, o4シリーズ、GPT-5シリーズ、GPT-4.1シリーズ
    """
    return model.startswith(("o1", "o3", "o4", "gpt-5", "gpt-4.1"))


def call_openai(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    json_output: bool,
) -> ChatResponse:
    """OpenAI API呼び出し"""
    client = get_openai_client()

    kwargs = {
        "model": model,
        "messages": messages,
    }

    # 推論モデル（o1, o3, o4）は max_completion_tokens を使用
    if _uses_max_completion_tokens(model):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature

    if json_output:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
    except OpenAIRateLimitError as e:
        raise LLMRateLimitError(str(e), provider="openai")
    except OpenAIAPIError as e:
        raise LLMAPIError(str(e), provider="openai", status_code=getattr(e, 'status_code', None))

    usage = None
    if response.usage:
        usage = Usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    return ChatResponse(
        content=response.choices[0].message.content,
        provider="openai",
        model=model,
        usage=usage,
        raw_response=response,
    )


def stream_openai(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> Iterator[StreamChunk]:
    """OpenAI APIストリーム呼び出し"""
    client = get_openai_client()

    kwargs = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    # 推論モデル（o1, o3, o4）は max_completion_tokens を使用
    if _uses_max_completion_tokens(model):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature

    try:
        stream = client.chat.completions.create(**kwargs)
    except OpenAIRateLimitError as e:
        raise LLMRateLimitError(str(e), provider="openai")
    except OpenAIAPIError as e:
        raise LLMAPIError(str(e), provider="openai", status_code=getattr(e, 'status_code', None))

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield StreamChunk(
                content=chunk.choices[0].delta.content,
                provider="openai",
                model=model,
            )
        # 最後のチャンクにusageが含まれる
        if chunk.usage:
            yield StreamChunk(
                content="",
                provider="openai",
                model=model,
                is_final=True,
                usage=Usage(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                ),
            )
