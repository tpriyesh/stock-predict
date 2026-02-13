"""
LLM provider plugins.
"""
import os
from typing import List

from loguru import logger

from providers.base import BaseLLMProvider


def get_llm_providers() -> List[BaseLLMProvider]:
    """Return available LLM providers in priority order."""
    from providers.llm.openai_provider import OpenAIProvider

    registry = {
        "openai": OpenAIProvider,
    }

    order_str = os.getenv("LLM_PROVIDERS", "openai")
    ordered_names = [s.strip().lower() for s in order_str.split(",") if s.strip()]

    providers = []
    for name in ordered_names:
        cls = registry.get(name)
        if cls is None:
            logger.warning(f"Unknown LLM provider '{name}', skipping")
            continue
        try:
            provider = cls()
            if provider.is_available():
                providers.append(provider)
                logger.info(f"LLM provider '{name}' registered")
        except Exception as e:
            logger.error(f"Failed to init LLM provider '{name}': {e}")

    return providers
