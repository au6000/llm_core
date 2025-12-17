"""
ユーティリティ関数（認証、モデル取得、JSONパース）
"""

import os
import json
from typing import Optional
from dotenv import load_dotenv

from .exceptions import LLMConfigError

# .envファイルを自動読み込み
load_dotenv()


# =============================================================================
# JSONパース
# =============================================================================
def extract_json(content: str) -> str:
    """
    レスポンスからJSONを抽出（マークダウンコードブロックを除去）

    対応フォーマット:
    - 純粋なJSON: {"key": "value"}
    - コードブロック: ```json\n{"key": "value"}\n```
    - コードブロック(言語指定なし): ```\n{"key": "value"}\n```
    """
    content = content.strip()

    if content.startswith("```"):
        lines = content.split("\n")
        lines = lines[1:]  # 最初の行を除去
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    return content.strip()


def parse_json(content: str) -> dict:
    """
    LLM出力からJSONをパースする

    Args:
        content: LLMの出力テキスト

    Returns:
        パースされた辞書

    Raises:
        json.JSONDecodeError: JSONのパースに失敗した場合
    """
    return json.loads(extract_json(content))


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

    認証方式（優先順位）:
        1. GOOGLE_APPLICATION_CREDENTIALS: サービスアカウントJSONファイルのパス
        2. Application Default Credentials (ADC): gcloud auth application-default login

    環境変数:
        - GOOGLE_APPLICATION_CREDENTIALS: サービスアカウントJSONファイルのパス（任意）
        - VERTEX_PROJECT_ID: GCPプロジェクトID（必須、または GOOGLE_CLOUD_PROJECT）
        - VERTEX_LOCATION: リージョン（デフォルト: us-central1）
    """
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    auth_method = None

    # 認証方式1: サービスアカウントJSONファイル
    if credentials_path:
        if not os.path.exists(credentials_path):
            raise LLMConfigError(
                f"Service account JSON file not found: {credentials_path}"
            )
        auth_method = "service_account"
    else:
        # 認証方式2: Application Default Credentials (ADC)
        auth_method = "adc"

    # プロジェクトIDの取得（優先順位: VERTEX_PROJECT_ID > GOOGLE_CLOUD_PROJECT > JSONから取得）
    project_id = os.environ.get("VERTEX_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")

    if not project_id:
        if credentials_path:
            # サービスアカウントJSONからプロジェクトIDを取得
            try:
                with open(credentials_path, "r") as f:
                    creds_data = json.load(f)
                    project_id = creds_data.get("project_id")
            except (json.JSONDecodeError, IOError) as e:
                raise LLMConfigError(f"Failed to read service account JSON: {e}")

        if not project_id:
            # ADC使用時はgcloudの設定からproject_idを取得試行
            try:
                import google.auth
                _, adc_project = google.auth.default()
                project_id = adc_project
            except Exception:
                pass

    if not project_id:
        raise LLMConfigError(
            "VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT not found. "
            "Set the environment variable or ensure gcloud is configured with a default project."
        )

    location = os.environ.get("VERTEX_LOCATION", "global")

    return {
        "credentials_path": credentials_path,  # Noneの場合はADCを使用
        "project_id": project_id,
        "location": location,
        "auth_method": auth_method,
    }


# =============================================================================
# モデル・料金取得
# =============================================================================
def get_model(provider: str, model: Optional[str] = None) -> str:
    """モデル名を取得（指定がなければデフォルト）"""
    from .config import DEFAULT_MODELS

    if model:
        return model

    default = DEFAULT_MODELS.get(provider)
    if not default:
        raise LLMConfigError(f"No default model for provider: {provider}")

    return default


def get_pricing(provider: str, model: str) -> dict:
    """料金情報を取得"""
    from .config import PRICING

    # vertexはgoogleと同じ料金体系
    pricing_key = "google" if provider == "vertex" else provider
    provider_pricing = PRICING.get(pricing_key, {})
    return provider_pricing.get(model, {"input": 0, "output": 0})
