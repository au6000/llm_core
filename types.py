"""
LLM関連の型定義
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum


class Provider(str, Enum):
    """LLMプロバイダー"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class ChatMessage:
    """チャットメッセージ"""
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class Usage:
    """トークン使用量"""
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ChatResponse:
    """LLMレスポンス"""
    content: str
    provider: str
    model: str
    usage: Optional[Usage] = None
    raw_response: Optional[object] = None  # デバッグ用


@dataclass
class ChatRequest:
    """LLMリクエスト設定"""
    prompt: str
    system_prompt: Optional[str] = None
    messages: Optional[list[ChatMessage]] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.0
    json_output: bool = False
    track_usage: bool = True
    retry_count: int = 3
    retry_delay: float = 1.0


@dataclass
class StreamChunk:
    """ストリームのチャンク"""
    content: str  # このチャンクのテキスト
    provider: str
    model: str
    is_final: bool = False  # 最後のチャンクかどうか
    usage: Optional[Usage] = None  # 最後のチャンクにのみ含まれる
