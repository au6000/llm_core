"""
LLMクライアント - 各プロバイダーとの通信を担当
"""

import json
import time
from functools import lru_cache
from typing import Optional, Union, Iterator

from openai import OpenAI, APIError as OpenAIAPIError, RateLimitError as OpenAIRateLimitError
from anthropic import Anthropic, APIError as AnthropicAPIError, RateLimitError as AnthropicRateLimitError
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted as GoogleRateLimitError

from .config import get_api_key, get_model, get_vertex_config, DEFAULT_PROVIDER
from .types import ChatRequest, ChatResponse, StreamChunk, Usage
from .exceptions import LLMAPIError, LLMRateLimitError, LLMResponseError, LLMConfigError
from . import usage_tracker


# =============================================================================
# クライアント取得（シングルトン）
# =============================================================================
@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI:
    return OpenAI(api_key=get_api_key("openai"))


@lru_cache(maxsize=1)
def _get_anthropic_client() -> Anthropic:
    return Anthropic(api_key=get_api_key("anthropic"))


@lru_cache(maxsize=1)
def _get_gemini_configured() -> bool:
    genai.configure(api_key=get_api_key("google"))
    return True


@lru_cache(maxsize=1)
def _get_vertex_client():
    """Vertex AI GenerativeModel用の設定を初期化"""
    import vertexai
    from google.oauth2 import service_account

    config = get_vertex_config()
    credentials = service_account.Credentials.from_service_account_file(
        config["credentials_path"]
    )
    vertexai.init(
        project=config["project_id"],
        location=config["location"],
        credentials=credentials,
    )
    return config


# =============================================================================
# プロバイダー別実装
# =============================================================================
def _call_openai(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    json_output: bool,
) -> ChatResponse:
    """OpenAI API呼び出し"""
    client = _get_openai_client()

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
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


def _call_anthropic(
    messages: list[dict],
    system_prompt: Optional[str],
    model: str,
    max_tokens: int,
    temperature: float,
    json_output: bool,
) -> ChatResponse:
    """Anthropic API呼び出し"""
    client = _get_anthropic_client()

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


def _call_google(
    messages: list[dict],
    system_prompt: Optional[str],
    model: str,
    max_tokens: int,
    temperature: float,
    json_output: bool,
) -> ChatResponse:
    """Google Gemini API呼び出し"""
    _get_gemini_configured()

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


def _call_vertex(
    messages: list[dict],
    system_prompt: Optional[str],
    model: str,
    max_tokens: int,
    temperature: float,
    json_output: bool,
) -> ChatResponse:
    """Vertex AI (Gemini) API呼び出し"""
    from vertexai.generative_models import GenerativeModel, GenerationConfig

    _get_vertex_client()

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


# =============================================================================
# プロバイダー別ストリーム実装
# =============================================================================
def _stream_openai(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> Iterator[StreamChunk]:
    """OpenAI APIストリーム呼び出し"""
    client = _get_openai_client()

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
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


def _stream_anthropic(
    messages: list[dict],
    system_prompt: Optional[str],
    model: str,
    max_tokens: int,
    temperature: float,
) -> Iterator[StreamChunk]:
    """Anthropic APIストリーム呼び出し"""
    client = _get_anthropic_client()

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


def _stream_google(
    messages: list[dict],
    system_prompt: Optional[str],
    model: str,
    max_tokens: int,
    temperature: float,
) -> Iterator[StreamChunk]:
    """Google Gemini APIストリーム呼び出し"""
    _get_gemini_configured()

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


def _stream_vertex(
    messages: list[dict],
    system_prompt: Optional[str],
    model: str,
    max_tokens: int,
    temperature: float,
) -> Iterator[StreamChunk]:
    """Vertex AI (Gemini) APIストリーム呼び出し"""
    from vertexai.generative_models import GenerativeModel, GenerationConfig

    _get_vertex_client()

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


# =============================================================================
# メイン関数
# =============================================================================
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
                response = _call_openai(msg_list, model, request.max_tokens, request.temperature, request.json_output)

            elif provider == "anthropic":
                response = _call_anthropic(msg_list, request.system_prompt, model, request.max_tokens, request.temperature, request.json_output)

            elif provider == "google":
                response = _call_google(msg_list, request.system_prompt, model, request.max_tokens, request.temperature, request.json_output)

            elif provider == "vertex":
                response = _call_vertex(msg_list, request.system_prompt, model, request.max_tokens, request.temperature, request.json_output)

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
        stream = _stream_openai(msg_list, model, request.max_tokens, request.temperature)

    elif provider == "anthropic":
        stream = _stream_anthropic(msg_list, request.system_prompt, model, request.max_tokens, request.temperature)

    elif provider == "google":
        stream = _stream_google(msg_list, request.system_prompt, model, request.max_tokens, request.temperature)

    elif provider == "vertex":
        stream = _stream_vertex(msg_list, request.system_prompt, model, request.max_tokens, request.temperature)

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
