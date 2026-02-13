"""
OpenAI LLM provider plugin.

Rate limiting: 180 RPM default (gpt-4o-mini tier).
Cost tracking: ~$0.15/1K input tokens, ~$0.60/1K output tokens for gpt-4o-mini.
Blended estimate: ~$0.0004/1K tokens (mostly short prompts + responses).
Daily cost cap: $5/day default (env-configurable).
"""
import os
from typing import Dict, List, Optional

from loguru import logger

from providers.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT — chat completions with cost tracking."""

    def __init__(self):
        self._api_key = os.getenv("OPENAI_API_KEY", "")
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def cost_per_1k_tokens(self) -> float:
        """Blended cost estimate for gpt-4o-mini."""
        return float(os.getenv("OPENAI_COST_PER_1K", "0.0004"))

    @property
    def requests_per_minute(self) -> int:
        return int(os.getenv("OPENAI_RPM", "180"))

    @property
    def daily_cost_cap(self) -> float:
        return float(os.getenv("OPENAI_DAILY_COST_CAP", "5.0"))

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _get_client(self):
        """Lazy-init OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def complete(self, messages: List[Dict[str, str]],
                 temperature: float = 0.1,
                 max_tokens: int = 500) -> Optional[str]:
        """Chat completion via OpenAI API.

        Returns response text or None on failure.
        Raises on rate limit (429) so quota manager can track it.
        """
        client = self._get_client()

        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0] if response.choices else None
        if not choice:
            return None

        return choice.message.content.strip()

    def estimate_cost(self, messages: List[Dict[str, str]],
                      max_tokens: int = 500) -> float:
        """Estimate cost of a completion call (rough approximation)."""
        # Rough token estimate: 4 chars per token
        total_chars = sum(len(m.get("content", "")) for m in messages)
        input_tokens = total_chars / 4
        total_tokens = input_tokens + max_tokens
        return (total_tokens / 1000) * self.cost_per_1k_tokens
