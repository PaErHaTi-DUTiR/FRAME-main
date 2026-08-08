from __future__ import annotations

import os
from typing import Any, Dict, Optional

from utils.cache import SimpleFileCache

from .base_backend import BaseLLMBackend
from .mock_backend import MockLLMBackend
from .openai_compatible import OpenAICompatibleBackend

_PROVIDER_DEFAULT_BASE_URL = {
    "openai": "https://api.openai.com/v1",
    "gpt-4": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

_PROVIDER_DEFAULT_MODEL = {
    "openai": "gpt-4o-mini",
    "gpt-4": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "qwen": "qwen-max",
}

_PROVIDER_ENV = {
    "openai": {
        "api_key": "OPENAI_API_KEY",
        "base_url": "OPENAI_BASE_URL",
        "model_name": "OPENAI_MODEL_NAME",
    },
    "gpt-4": {
        "api_key": "OPENAI_API_KEY",
        "base_url": "OPENAI_BASE_URL",
        "model_name": "OPENAI_MODEL_NAME",
    },
    "deepseek": {
        "api_key": "DEEPSEEK_API_KEY",
        "base_url": "DEEPSEEK_BASE_URL",
        "model_name": "DEEPSEEK_MODEL_NAME",
    },
    "qwen": {
        "api_key": "QWEN_API_KEY",
        "base_url": "QWEN_BASE_URL",
        "model_name": "QWEN_MODEL_NAME",
    },
}


def _resolve_provider_value(provider: str, config: Dict[str, Any], key: str) -> str:
    env_table = _PROVIDER_ENV.get(provider, {})
    env_name_key = f"{key}_env"
    explicit_env_name = str(config.get(env_name_key, "")).strip()
    env_name = explicit_env_name or env_table.get(key, "")

    # If an explicit env var is configured, prioritize it over inherited defaults
    # from merged configs (e.g., global mock model_name).
    if explicit_env_name:
        explicit_env_value = os.getenv(explicit_env_name, "").strip()
        if explicit_env_value:
            return explicit_env_value

    cfg_value = str(config.get(key, "")).strip()
    if cfg_value:
        return cfg_value

    if env_name:
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            return env_value

    if key == "base_url":
        return str(_PROVIDER_DEFAULT_BASE_URL.get(provider, "https://api.openai.com/v1"))
    if key == "model_name":
        return str(_PROVIDER_DEFAULT_MODEL.get(provider, "gpt-4o-mini"))
    return ""


def _build_openai_compatible_backend(provider_key: str, config: Dict[str, Any]) -> OpenAICompatibleBackend:
    model_name = _resolve_provider_value(provider_key, config, "model_name")
    base_url = _resolve_provider_value(provider_key, config, "base_url")
    api_key = _resolve_provider_value(provider_key, config, "api_key")

    temperature = float(config.get("temperature", 0.0))
    max_tokens = int(config.get("max_tokens", 1024))
    timeout = int(config.get("timeout", 60))
    max_retries = int(config.get("max_retries", 3))
    retry_backoff_sec = float(config.get("retry_backoff_sec", 1.0))
    retry_max_wait_sec = float(config.get("retry_max_wait_sec", 8.0))
    enable_response_logging = bool(config.get("enable_response_logging", True))

    cache = None
    if bool(config.get("enable_cache", False)):
        cache_dir = str(config.get("cache_dir", "outputs/cache/llm"))
        cache = SimpleFileCache(cache_dir)
    cache_ttl_seconds = config.get("cache_ttl_seconds", 600)

    no_proxy = bool(config.get("no_proxy", False))
    extra_body = config.get("extra_body") if isinstance(config.get("extra_body"), dict) else None

    return OpenAICompatibleBackend(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff_sec=retry_backoff_sec,
        retry_max_wait_sec=retry_max_wait_sec,
        enable_response_logging=enable_response_logging,
        cache=cache,
        cache_ttl_seconds=cache_ttl_seconds,
        provider_name=provider_key,
        no_proxy=no_proxy,
        extra_body=extra_body,
    )


def create_backend(provider: str, config: Dict[str, Any]) -> BaseLLMBackend:
    provider_key = provider.lower()

    if provider_key in {"mock", "mock_llm"}:
        return MockLLMBackend(
            model_name=str(config.get("model_name", "mock-llm")),
            deterministic=bool(config.get("deterministic", True)),
            seed=int(config.get("seed", 42)),
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 1024)),
        )

    if provider_key in {"openai", "gpt-4", "deepseek", "qwen", "openai_compatible"}:
        compatible_provider = provider_key if provider_key != "openai_compatible" else "openai"
        return _build_openai_compatible_backend(compatible_provider, config)

    raise ValueError(f"Unsupported backend provider: {provider}")


def create_backend_from_config(config: Optional[Dict[str, Any]]) -> Optional[BaseLLMBackend]:
    if not config:
        return None

    mode = str(config.get("mode", "mock")).strip().lower()
    if mode == "off":
        return None

    if mode == "mock":
        mock_cfg = dict(config)
        provider = str(mock_cfg.get("provider", "mock"))
        return create_backend(provider, mock_cfg)

    if mode == "real":
        real_cfg = dict(config)
        provider = str(real_cfg.get("provider", "openai_compatible"))
        return create_backend(provider, real_cfg)

    raise ValueError(f"Unsupported backend mode: {mode}")
