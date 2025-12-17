"""
Google Gemini プロバイダー実装
"""

from typing import Iterator, Optional

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted as GoogleRateLimitError

from ..types import ChatResponse, StreamChunk, Usage
from ..exceptions import LLMAPIError, LLMRateLimitError
from .._init_clients import get_gemini_configured


def call_google(
    messages: list[dict],
    system_prompt: Optional[str],
    model: str,
    max_tokens: int,
    temperature: float,
    json_output: bool,
) -> ChatResponse:
    """Google Gemini API呼び出し"""
    get_gemini_configured()

    # Gemini用にメッセージを変換
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    generation_config = {
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_output:
        generation_config["response_mime_type"] = "application/json"

    model_instance = genai.GenerativeModel(
        model,
        system_instruction=system_prompt if system_prompt else None,
    )

    try:
        response = model_instance.generate_content(
            contents,
            generation_config=generation_config,
        )
    except GoogleRateLimitError as e:
        raise LLMRateLimitError(str(e), provider="google")
    except Exception as e:
        raise LLMAPIError(str(e), provider="google")

    usage = None
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage = Usage(
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
        )

    return ChatResponse(
        content=response.text,
        provider="google",
        model=model,
        usage=usage,
        raw_response=response,
    )


def stream_google(
    messages: list[dict],
    system_prompt: Optional[str],
    model: str,
    max_tokens: int,
    temperature: float,
) -> Iterator[StreamChunk]:
    """Google Gemini APIストリーム呼び出し"""
    get_gemini_configured()

    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    generation_config = {
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }

    model_instance = genai.GenerativeModel(
        model,
        system_instruction=system_prompt if system_prompt else None,
    )

    try:
        response = model_instance.generate_content(
            contents,
            generation_config=generation_config,
            stream=True,
        )
        for chunk in response:
            if chunk.text:
                yield StreamChunk(
                    content=chunk.text,
                    provider="google",
                    model=model,
                )
        # Geminiは最後にusage_metadataを取得
        # NOTE: ストリーム時のusage取得はAPIの制限により不正確な場合がある
        yield StreamChunk(
            content="",
            provider="google",
            model=model,
            is_final=True,
        )
    except GoogleRateLimitError as e:
        raise LLMRateLimitError(str(e), provider="google")
    except Exception as e:
        raise LLMAPIError(str(e), provider="google")
