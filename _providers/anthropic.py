"""
Anthropic プロバイダー実装
"""

from typing import Iterator, Optional

from anthropic import APIError as AnthropicAPIError, RateLimitError as AnthropicRateLimitError

from ..types import ChatResponse, StreamChunk, Usage
from ..exceptions import LLMAPIError, LLMRateLimitError
from .._init_clients import get_anthropic_client


def call_anthropic(
    messages: list[dict],
    system_prompt: Optional[str],
    model: str,
    max_tokens: int,
    temperature: float,
    json_output: bool,
) -> ChatResponse:
    """Anthropic API呼び出し"""
    client = get_anthropic_client()

    system = system_prompt or ""
    if json_output and "JSON" not in system:
        system = (system + "\n" if system else "") + "必ずJSON形式で回答してください。"

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system.strip() if system else "",
            messages=messages,
        )
    except AnthropicRateLimitError as e:
        raise LLMRateLimitError(str(e), provider="anthropic")
    except AnthropicAPIError as e:
        raise LLMAPIError(str(e), provider="anthropic", status_code=getattr(e, 'status_code', None))

    usage = None
    if response.usage:
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    return ChatResponse(
        content=response.content[0].text,
        provider="anthropic",
        model=model,
        usage=usage,
        raw_response=response,
    )


def stream_anthropic(
    messages: list[dict],
    system_prompt: Optional[str],
    model: str,
    max_tokens: int,
    temperature: float,
) -> Iterator[StreamChunk]:
    """Anthropic APIストリーム呼び出し"""
    client = get_anthropic_client()

    try:
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield StreamChunk(
                    content=text,
                    provider="anthropic",
                    model=model,
                )
            # 最後にusageを取得
            response = stream.get_final_message()
            if response.usage:
                yield StreamChunk(
                    content="",
                    provider="anthropic",
                    model=model,
                    is_final=True,
                    usage=Usage(
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                    ),
                )
    except AnthropicRateLimitError as e:
        raise LLMRateLimitError(str(e), provider="anthropic")
    except AnthropicAPIError as e:
        raise LLMAPIError(str(e), provider="anthropic", status_code=getattr(e, 'status_code', None))
