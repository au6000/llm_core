"""
llm_core - 汎用LLMクライアントライブラリ

Usage:
    from llm_core import chat_completion, chat_completion_json, tracker

    # シンプルな呼び出し
    response = chat_completion("Hello!")
    print(response.content)

    # JSON出力
    data = chat_completion_json("Return a JSON with name and age")

    # モデル・プロバイダー指定
    response = chat_completion(
        "Explain quantum computing",
        model="gpt-4o",
        provider="openai",
        system_prompt="You are a helpful assistant.",
        max_tokens=1000,
        temperature=0.7,
    )

    # 使用量確認
    tracker.print_summary()
    stats = tracker.get_usage_stats()

    # 使用量保存
    tracker.save_usage("usage.json")

Required environment variables:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    - GOOGLE_API_KEY
"""

from .client import (
    chat_completion,
    chat_completion_json,
    chat_completion_text,
)

from .types import (
    ChatMessage,
    ChatResponse,
    Usage,
    Provider,
)

from .exceptions import (
    LLMError,
    LLMConfigError,
    LLMAPIError,
    LLMRateLimitError,
    LLMResponseError,
)

from . import usage_tracker as tracker

from .config import (
    PRICING,
    DEFAULT_MODELS,
    DEFAULT_PROVIDER,
)

__all__ = [
    # Main functions
    "chat_completion",
    "chat_completion_json",
    "chat_completion_text",
    # Types
    "ChatMessage",
    "ChatResponse",
    "Usage",
    "Provider",
    # Exceptions
    "LLMError",
    "LLMConfigError",
    "LLMAPIError",
    "LLMRateLimitError",
    "LLMResponseError",
    # Tracker
    "tracker",
    # Config
    "PRICING",
    "DEFAULT_MODELS",
    "DEFAULT_PROVIDER",
]

__version__ = "1.0.0"
