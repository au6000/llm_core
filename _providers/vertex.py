"""
Vertex AI (Gemini) プロバイダー実装
"""

from typing import Iterator, Optional

from google.api_core.exceptions import ResourceExhausted as GoogleRateLimitError

from ..types import ChatResponse, StreamChunk, Usage
from ..exceptions import LLMAPIError, LLMRateLimitError
from .._init_clients import get_vertex_client


def call_vertex(
    messages: list[dict],
    system_prompt: Optional[str],
    model: str,
    max_tokens: int,
    temperature: float,
    json_output: bool,
) -> ChatResponse:
    """Vertex AI (Gemini) API呼び出し"""
    from vertexai.generative_models import GenerativeModel, GenerationConfig

    get_vertex_client()

    # Vertex AI用にメッセージを変換
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    generation_config = GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    if json_output:
        generation_config = GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            response_mime_type="application/json",
        )

    model_instance = GenerativeModel(
        model,
        system_instruction=system_prompt if system_prompt else None,
    )

    try:
        response = model_instance.generate_content(
            contents,
            generation_config=generation_config,
        )
    except GoogleRateLimitError as e:
        raise LLMRateLimitError(str(e), provider="vertex")
    except Exception as e:
        raise LLMAPIError(str(e), provider="vertex")

    usage = None
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage = Usage(
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
        )

    return ChatResponse(
        content=response.text,
        provider="vertex",
        model=model,
        usage=usage,
        raw_response=response,
    )


def stream_vertex(
    messages: list[dict],
    system_prompt: Optional[str],
    model: str,
    max_tokens: int,
    temperature: float,
) -> Iterator[StreamChunk]:
    """Vertex AI (Gemini) APIストリーム呼び出し"""
    from vertexai.generative_models import GenerativeModel, GenerationConfig

    get_vertex_client()

    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    generation_config = GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )

    model_instance = GenerativeModel(
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
                    provider="vertex",
                    model=model,
                )
        yield StreamChunk(
            content="",
            provider="vertex",
            model=model,
            is_final=True,
        )
    except GoogleRateLimitError as e:
        raise LLMRateLimitError(str(e), provider="vertex")
    except Exception as e:
        raise LLMAPIError(str(e), provider="vertex")
