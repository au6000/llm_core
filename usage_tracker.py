"""
LLMの使用量・コストを記録・集計するモジュール
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from .config import get_pricing, USD_TO_JPY


@dataclass
class UsageRecord:
    """1回のAPI呼び出し記録"""
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class UsageTracker:
    """トークン使用量とコストを記録"""

    def __init__(self):
        self._records: list[UsageRecord] = []
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost_usd: float = 0.0

    def add(self, provider: str, model: str, input_tokens: int, output_tokens: int):
        """使用量を追加"""
        pricing = get_pricing(provider, model)
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        record = UsageRecord(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

        self._records.append(record)
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost_usd += cost

    @property
    def call_count(self) -> int:
        return len(self._records)

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    @property
    def total_cost_jpy(self) -> float:
        return self._total_cost_usd * USD_TO_JPY

    def summary(self) -> dict:
        """サマリーを返す"""
        return {
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_cost_jpy": round(self.total_cost_jpy, 2),
        }

    def details(self) -> list[dict]:
        """詳細な使用量履歴を取得"""
        return [
            {
                "provider": r.provider,
                "model": r.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": r.cost_usd,
                "timestamp": r.timestamp,
            }
            for r in self._records
        ]

    def reset(self):
        """統計をリセット"""
        self._records = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost_usd = 0.0

    def print_summary(self):
        """使用量サマリーを出力"""
        print(f"\n{'='*40}")
        print("LLM Usage Summary")
        print('='*40)
        print(f"API呼び出し回数: {self.call_count}")
        print(f"入力トークン:    {self.total_input_tokens:,}")
        print(f"出力トークン:    {self.total_output_tokens:,}")
        print(f"合計トークン:    {self.total_tokens:,}")
        print(f"推定コスト:      ${self.total_cost_usd:.4f} (約{self.total_cost_jpy:.0f}円)")
        print('='*40 + "\n")

    def save_to_file(self, filepath: str):
        """使用量をJSONファイルに保存"""
        data = {
            "summary": self.summary(),
            "details": self.details(),
        }
        Path(filepath).write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def load_from_file(self, filepath: str):
        """JSONファイルから使用量を読み込み（累積）"""
        data = json.loads(Path(filepath).read_text())
        for record in data.get("details", []):
            self.add(
                provider=record["provider"],
                model=record["model"],
                input_tokens=record["input_tokens"],
                output_tokens=record["output_tokens"],
            )


# =============================================================================
# グローバルインスタンス
# =============================================================================
_default_tracker = UsageTracker()


def get_default_tracker() -> UsageTracker:
    """デフォルトトラッカーを取得"""
    return _default_tracker


def add_usage(provider: str, model: str, input_tokens: int, output_tokens: int):
    """使用量を追加（デフォルトトラッカー）"""
    _default_tracker.add(provider, model, input_tokens, output_tokens)


def get_usage_stats() -> dict:
    """現在の使用量統計を取得"""
    return _default_tracker.summary()


def get_usage_details() -> list[dict]:
    """詳細な使用量履歴を取得"""
    return _default_tracker.details()


def reset_usage_stats():
    """使用量統計をリセット"""
    _default_tracker.reset()


def print_usage_summary():
    """使用量サマリーを出力"""
    _default_tracker.print_summary()


def save_usage(filepath: str):
    """使用量をファイルに保存"""
    _default_tracker.save_to_file(filepath)


def estimate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    """
    コストを事前見積もり

    Returns:
        dict: {
            "input_tokens": int,
            "output_tokens": int,
            "total_tokens": int,
            "cost_usd": float,
            "cost_jpy": float,
        }
    """
    pricing = get_pricing(provider, model)
    cost_usd = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    cost_jpy = cost_usd * USD_TO_JPY

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": round(cost_usd, 6),
        "cost_jpy": round(cost_jpy, 2),
    }


def estimate_batch_cost(
    provider: str,
    model: str,
    num_requests: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
) -> dict:
    """
    バッチ処理のコストを事前見積もり

    Args:
        provider: プロバイダー名
        model: モデル名
        num_requests: リクエスト数
        avg_input_tokens: 1リクエストあたりの平均入力トークン数
        avg_output_tokens: 1リクエストあたりの平均出力トークン数

    Returns:
        dict: 見積もり結果
    """
    total_input = num_requests * avg_input_tokens
    total_output = num_requests * avg_output_tokens

    result = estimate_cost(provider, model, total_input, total_output)
    result["num_requests"] = num_requests
    result["avg_input_tokens"] = avg_input_tokens
    result["avg_output_tokens"] = avg_output_tokens

    return result


def print_estimate(estimate: dict):
    """見積もり結果を出力"""
    print(f"\n{'='*40}")
    print("Cost Estimate (見積もり)")
    print('='*40)
    if "num_requests" in estimate:
        print(f"リクエスト数:    {estimate['num_requests']:,}")
        print(f"平均入力トークン: {estimate['avg_input_tokens']:,}")
        print(f"平均出力トークン: {estimate['avg_output_tokens']:,}")
    print(f"合計入力トークン: {estimate['input_tokens']:,}")
    print(f"合計出力トークン: {estimate['output_tokens']:,}")
    print(f"合計トークン:    {estimate['total_tokens']:,}")
    print(f"推定コスト:      ${estimate['cost_usd']:.4f} (約{estimate['cost_jpy']:.0f}円)")
    print('='*40 + "\n")
