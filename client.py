"""
LLMクライアント - 各プロバイダーとの通信を担当
"""

import json
import time
from functools import lru_cache
from typing import Optional, Union

from openai import OpenAI, APIError as OpenAIAPIError, RateLimitError as OpenAIRateLimitError
from anthropic import Anthropic, APIError as AnthropicAPIError, RateLimitError as AnthropicRateLimitError
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted as GoogleRateLimitError

from .config import get_api_key, get_model, get_vertex_config, DEFAULT_PROVIDER
from .types import ChatMessage, ChatResponse, Usage, Provider
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
# メイン関数
# =============================================================================
def chat_completion(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    messages: Optional[list[ChatMessage]] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    json_output: bool = False,
    track_usage: bool = True,
    retry_count: int = 3,
    retry_delay: float = 1.0,
) -> ChatResponse:
    """
    LLMにチャット補完リクエストを送信

    Args:
        prompt: ユーザープロンプト
        system_prompt: システムプロンプト（省略可）
        messages: メッセージ履歴（promptと併用不可、指定時はpromptは無視）
        model: モデル名（省略時はプロバイダーのデフォルト）
        provider: プロバイダー（"openai", "anthropic", "google", "vertex"）
        max_tokens: 最大出力トークン数
        temperature: 温度パラメータ
        json_output: JSON形式で出力を要求
        track_usage: 使用量を記録するか
        retry_count: リトライ回数
        retry_delay: リトライ間隔（秒）

    Returns:
        ChatResponse: レスポンス
    """
    provider = provider or DEFAULT_PROVIDER
    model = get_model(provider, model)

    # メッセージ構築
    if messages:
        msg_list = [{"role": m.role, "content": m.content} for m in messages]
    else:
        msg_list = [{"role": "user", "content": prompt}]

    # リトライループ
    last_error = None
    for attempt in range(retry_count):
        try:
            if provider == "openai":
                # OpenAIはsystem_promptをmessagesに含める
                if system_prompt:
                    msg_list = [{"role": "system", "content": system_prompt}] + msg_list
                response = _call_openai(msg_list, model, max_tokens, temperature, json_output)

            elif provider == "anthropic":
                response = _call_anthropic(msg_list, system_prompt, model, max_tokens, temperature, json_output)

            elif provider == "google":
                response = _call_google(msg_list, system_prompt, model, max_tokens, temperature, json_output)

            elif provider == "vertex":
                response = _call_vertex(msg_list, system_prompt, model, max_tokens, temperature, json_output)

            else:
                raise LLMConfigError(f"Unknown provider: {provider}")

            # 使用量記録
            if track_usage and response.usage:
                usage_tracker.add_usage(
                    provider=provider,
                    model=model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )

            return response

        except LLMRateLimitError as e:
            last_error = e
            if attempt < retry_count - 1:
                time.sleep(retry_delay * (attempt + 1))  # 指数バックオフ的
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


def chat_completion_json(
    prompt: str,
    **kwargs,
) -> dict:
    """
    JSON形式でレスポンスを取得し、dictにパースして返す

    Args:
        prompt: プロンプト
        **kwargs: chat_completionに渡す引数

    Returns:
        dict: パースされたJSON
    """
    kwargs["json_output"] = True
    response = chat_completion(prompt, **kwargs)

    try:
        content = _extract_json_from_response(response.content)
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise LLMResponseError(f"Failed to parse JSON response: {e}\nResponse: {response.content[:500]}")


def chat_completion_text(
    prompt: str,
    **kwargs,
) -> str:
    """
    テキストレスポンスのみを返す（シンプルなラッパー）

    Args:
        prompt: プロンプト
        **kwargs: chat_completionに渡す引数

    Returns:
        str: レスポンステキスト
    """
    response = chat_completion(prompt, **kwargs)
    return response.content
