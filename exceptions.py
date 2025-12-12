"""
LLM関連のカスタム例外
"""


class LLMError(Exception):
    """LLM関連エラーの基底クラス"""
    pass


class LLMConfigError(LLMError):
    """設定エラー（API KEY未設定等）"""
    pass


class LLMAPIError(LLMError):
    """API呼び出しエラー"""
    def __init__(self, message: str, provider: str = None, status_code: int = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class LLMRateLimitError(LLMAPIError):
    """レート制限エラー"""
    pass


class LLMResponseError(LLMError):
    """レスポンス解析エラー（JSON parse失敗等）"""
    pass
