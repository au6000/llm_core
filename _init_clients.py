"""
LLMクライアント初期化（シングルトン）
"""

from functools import lru_cache

from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

from .utils import get_api_key, get_vertex_config


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """OpenAIクライアントを取得（シングルトン）"""
    return OpenAI(api_key=get_api_key("openai"))


@lru_cache(maxsize=1)
def get_anthropic_client() -> Anthropic:
    """Anthropicクライアントを取得（シングルトン）"""
    return Anthropic(api_key=get_api_key("anthropic"))


@lru_cache(maxsize=1)
def get_gemini_configured() -> bool:
    """Gemini APIを設定（シングルトン）"""
    genai.configure(api_key=get_api_key("google"))
    return True


@lru_cache(maxsize=1)
def get_vertex_client():
    """
    Vertex AI GenerativeModel用の設定を初期化

    認証方式:
        1. サービスアカウントJSON (GOOGLE_APPLICATION_CREDENTIALS設定時)
        2. Application Default Credentials (ADC) - gcloud auth application-default login
    """
    import vertexai
    import google.auth

    config = get_vertex_config()
    credentials = None

    if config["credentials_path"]:
        # サービスアカウントJSONから認証情報を取得
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(
            config["credentials_path"]
        )
    else:
        # Application Default Credentials (ADC) を使用
        credentials, _ = google.auth.default()

    vertexai.init(
        project=config["project_id"],
        location=config["location"],
        credentials=credentials,
    )
    return config
