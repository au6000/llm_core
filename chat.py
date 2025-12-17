"""
チャット補完メイン関数
"""

import json
import time
from typing import Iterator

from .config import get_model, DEFAULT_PROVIDER
from .types import ChatRequest, ChatResponse, StreamChunk, Usage
from .exceptions import LLMConfigError, LLMRateLimitError, LLMAPIError, LLMResponseError
from . import usage_tracker
from ._providers import (
    call_openai,
    stream_openai,
    call_anthropic,
    stream_anthropic,
    call_google,
    stream_google,
    call_vertex,
    stream_vertex,
)


def chat_completion(request: ChatRequest) -> ChatResponse:
    """
    LLMにチャット補完リクエストを送信

    Args:
        request: ChatRequestオブジェクト

    Returns:
        ChatResponse: レスポンス
    """
    provider = request.provider or DEFAULT_PROVIDER
    model = get_model(provider, request.model)

    # メッセージ構築
    if request.messages:
        msg_list = [{"role": m.role, "content": m.content} for m in request.messages]
    else:
        msg_list = [{"role": "user", "content": request.prompt}]

    # リトライループ
    last_error = None
    for attempt in range(request.retry_count):
        try:
            if provider == "openai":
                # OpenAIはsystem_promptをmessagesに含める
                if request.system_prompt:
                    msg_list = [{"role": "system", "content": request.system_prompt}] + msg_list
                response = call_openai(msg_list, model, request.max_tokens, request.temperature, request.json_output)

            elif provider == "anthropic":
                response = call_anthropic(msg_list, request.system_prompt, model, request.max_tokens, request.temperature, request.json_output)

            elif provider == "google":
                response = call_google(msg_list, request.system_prompt, model, request.max_tokens, request.temperature, request.json_output)

            elif provider == "vertex":
                response = call_vertex(msg_list, request.system_prompt, model, request.max_tokens, request.temperature, request.json_output)

            else:
                raise LLMConfigError(f"Unknown provider: {provider}")

            # 使用量記録
            if request.track_usage and response.usage:
                usage_tracker.add_usage(
                    provider=provider,
                    model=model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )

            return response

        except LLMRateLimitError as e:
            last_error = e
            if attempt < request.retry_count - 1:
                time.sleep(request.retry_delay * (attempt + 1))  # 指数バックオフ的
                continue
            raise

        except LLMAPIError:
            raise

    raise last_error


def _extract_json_from_response(content: str) -> str:
    """
    レスポンスからJSONを抽出（マークダウンコードブロックを除去）
    """
    content = content.strip()

    # ```json ... ``` または ``` ... ``` 形式の場合
    if content.startswith("```"):
        lines = content.split("\n")
        # 最初の行（```json など）を除去
        if lines[0].startswith("```"):
            lines = lines[1:]
        # 最後の行（```）を除去
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    return content.strip()


def chat_completion_json(request: ChatRequest) -> dict:
    """
    JSON形式でレスポンスを取得し、dictにパースして返す

    Args:
        request: ChatRequestオブジェクト（json_outputは自動でTrueに設定される）

    Returns:
        dict: パースされたJSON
    """
    request.json_output = True
    response = chat_completion(request)

    try:
        content = _extract_json_from_response(response.content)
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise LLMResponseError(f"Failed to parse JSON response: {e}\nResponse: {response.content[:500]}")


def chat_completion_text(request: ChatRequest) -> str:
    """
    テキストレスポンスのみを返す（シンプルなラッパー）

    Args:
        request: ChatRequestオブジェクト

    Returns:
        str: レスポンステキスト
    """
    response = chat_completion(request)
    return response.content


def chat_completion_stream(request: ChatRequest) -> Iterator[StreamChunk]:
    """
    LLMにストリーミングリクエストを送信

    Args:
        request: ChatRequestオブジェクト

    Yields:
        StreamChunk: ストリームのチャンク

    Usage:
        request = ChatRequest(prompt="Hello!")
        for chunk in chat_completion_stream(request):
            print(chunk.content, end="", flush=True)
            if chunk.is_final and chunk.usage:
                print(f"\\nTokens: {chunk.usage.total_tokens}")
    """
    provider = request.provider or DEFAULT_PROVIDER
    model = get_model(provider, request.model)

    # メッセージ構築
    if request.messages:
        msg_list = [{"role": m.role, "content": m.content} for m in request.messages]
    else:
        msg_list = [{"role": "user", "content": request.prompt}]

    if provider == "openai":
        if request.system_prompt:
            msg_list = [{"role": "system", "content": request.system_prompt}] + msg_list
        stream = stream_openai(msg_list, model, request.max_tokens, request.temperature)

    elif provider == "anthropic":
        stream = stream_anthropic(msg_list, request.system_prompt, model, request.max_tokens, request.temperature)

    elif provider == "google":
        stream = stream_google(msg_list, request.system_prompt, model, request.max_tokens, request.temperature)

    elif provider == "vertex":
        stream = stream_vertex(msg_list, request.system_prompt, model, request.max_tokens, request.temperature)

    else:
        raise LLMConfigError(f"Unknown provider: {provider}")

    # 使用量記録用
    final_usage = None

    for chunk in stream:
        if chunk.is_final and chunk.usage:
            final_usage = chunk.usage
        yield chunk

    # 使用量記録
    if request.track_usage and final_usage:
        usage_tracker.add_usage(
            provider=provider,
            model=model,
            input_tokens=final_usage.input_tokens,
            output_tokens=final_usage.output_tokens,
        )
