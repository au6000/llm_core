"""
LLM設定（料金表、デフォルト設定）
環境変数からAPI KEYを読み込み
"""

import os
from dotenv import load_dotenv

# .envファイルを自動読み込み
load_dotenv()
from typing import Optional
from .exceptions import LLMConfigError


# =============================================================================
# 料金表 (USD per 1M tokens)
# =============================================================================
PRICING = {
    "openai": {
        # 新世代 GPT‑5 ファミリー
        "gpt-5.2": {"input": 1.75, "output": 14.00},
        "gpt-5.1": {"input": 1.25, "output": 10.00},
        "gpt-5": {"input": 1.25, "output": 10.00},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-5-nano": {"input": 0.05, "output": 0.40},
        "gpt-5.2-pro": {"input": 21.00, "output": 168.00},
        "gpt-5-pro": {"input": 15.00, "output": 120.00},

        # GPT‑4系およびその他
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-realtime": {"input": 4.00, "output": 16.00},
        "gpt-realtime-mini": {"input": 0.60, "output": 2.40},

        # oシリーズ
        "o1": {"input": 15.00, "output": 60.00},
        "o1-mini": {"input": 1.10, "output": 4.40},
        "o1-pro": {"input": 150.00, "output": 600.00},
        "o3": {"input": 2.00, "output": 8.00},
        "o3-pro": {"input": 20.00, "output": 80.00},
        "o3-mini": {"input": 1.10, "output": 4.40},
        "o4-mini": {"input": 1.10, "output": 4.40},
        "o4-mini-deep-research": {"input": 2.00, "output": 8.00},
    },
    "anthropic": {
        # Claude 4.5 シリーズ（API ID付き）
        "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00},    # Opus 4.5:contentReference[oaicite:1]{index=1}。
        "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},  # Sonnet 4.5:contentReference[oaicite:2]{index=2}。
        "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},    # Haiku 4.5:contentReference[oaicite:3]{index=3}。

        # Claude 4 系列（旧世代だがまだアクティブ）
        "claude-opus-4-1-20250805": {"input": 15.00, "output": 75.00},   # Opus 4.1:contentReference[oaicite:4]{index=4}。
        "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},     # Opus 4:contentReference[oaicite:5]{index=5}。
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},    # Sonnet 4:contentReference[oaicite:6]{index=6}。

        # Claude 3.x 系列（提供・デprecated）
        "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},  # Sonnet 3.7:contentReference[oaicite:7]{index=7}。
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},    # Haiku 3.5:contentReference[oaicite:8]{index=8}。
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},      # Haiku 3:contentReference[oaicite:9]{index=9}。
    },
    "google": {
        # Gemini 3 世代
        "gemini-3-pro": {"input": 2.00, "output": 12.00},
        # Gemini 2.5 世代
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
        "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
        "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
        # Gemini 2.0 世代
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    },
}



# =============================================================================
# デフォルトモデル
# =============================================================================
DEFAULT_MODELS = {
    "openai": "gpt-5",
    "anthropic": "claude-sonnet-4-5-20250929",
    "google": "gemini-3-pro",
    "vertex": "gemini-3-pro",
}


# =============================================================================
# デフォルトプロバイダー
# =============================================================================
DEFAULT_PROVIDER = "anthropic"


# =============================================================================
# API KEY取得
# =============================================================================
def get_api_key(provider: str) -> str:
    """
    環境変数からAPI KEYを取得

    環境変数名:
        - openai: OPENAI_API_KEY
        - anthropic: ANTHROPIC_API_KEY
        - google: GOOGLE_API_KEY
        - vertex: 不要（サービスアカウントJSONで認証）
    """
    env_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    # Vertexはサービスアカウント認証のためAPI KEYは不要
    if provider == "vertex":
        return ""

    env_var = env_vars.get(provider)
    if not env_var:
        raise LLMConfigError(f"Unknown provider: {provider}")

    api_key = os.environ.get(env_var)
    if not api_key:
        raise LLMConfigError(
            f"API KEY not found. Set environment variable: {env_var}"
        )

    return api_key


def get_vertex_config() -> dict:
    """
    Vertex AI用の設定を取得

    環境変数:
        - GOOGLE_APPLICATION_CREDENTIALS: サービスアカウントJSONファイルのパス
        - VERTEX_PROJECT_ID: GCPプロジェクトID（省略時はJSONから取得）
        - VERTEX_LOCATION: リージョン（デフォルト: us-central1）
    """
    import json as _json

    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise LLMConfigError(
            "GOOGLE_APPLICATION_CREDENTIALS not found. Set the path to your service account JSON file."
        )

    if not os.path.exists(credentials_path):
        raise LLMConfigError(
            f"Service account JSON file not found: {credentials_path}"
        )

    # プロジェクトIDをJSONから取得（環境変数で上書き可能）
    project_id = os.environ.get("VERTEX_PROJECT_ID")
    if not project_id:
        with open(credentials_path, "r") as f:
            creds_data = _json.load(f)
            project_id = creds_data.get("project_id")
        if not project_id:
            raise LLMConfigError(
                "VERTEX_PROJECT_ID not found in environment or service account JSON."
            )

    location = os.environ.get("VERTEX_LOCATION", "us-central1")

    return {
        "credentials_path": credentials_path,
        "project_id": project_id,
        "location": location,
    }


def get_model(provider: str, model: Optional[str] = None) -> str:
    """モデル名を取得（指定がなければデフォルト）"""
    if model:
        return model

    default = DEFAULT_MODELS.get(provider)
    if not default:
        raise LLMConfigError(f"No default model for provider: {provider}")

    return default


def get_pricing(provider: str, model: str) -> dict:
    """料金情報を取得"""
    provider_pricing = PRICING.get(provider, {})
    return provider_pricing.get(model, {"input": 0, "output": 0})


# =============================================================================
# 為替レート（概算）
# =============================================================================
USD_TO_JPY = 150.0
