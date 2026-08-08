from __future__ import annotations

import hashlib
import json
import time
from typing import Dict, List, Optional

import requests

from utils.cache import SimpleFileCache
from utils.json_parser import parse_json
from utils.logger import get_logger

from .base_backend import BaseLLMBackend, ChatMessage


class OpenAICompatibleBackend(BaseLLMBackend):
    """Backend for OpenAI-compatible chat completion endpoints.

    Works with OpenAI, DeepSeek, and Qwen-style OpenAI-compatible APIs.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 60,
        max_retries: int = 3,
        retry_backoff_sec: float = 1.0,
        retry_max_wait_sec: float = 8.0,
        enable_response_logging: bool = True,
        cache: Optional[SimpleFileCache] = None,
        cache_ttl_seconds: Optional[int] = 600,
        provider_name: str = "openai_compatible",
        no_proxy: bool = False,
        extra_body: Optional[Dict] = None,
    ):
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_sec = max(0.0, float(retry_backoff_sec))
        self.retry_max_wait_sec = max(0.0, float(retry_max_wait_sec))
        self.enable_response_logging = enable_response_logging
        self.cache = cache
        self.cache_ttl_seconds = cache_ttl_seconds
        self.provider_name = provider_name
        self.no_proxy = no_proxy
        self.extra_body = extra_body or {}
        self.logger = get_logger("karve.backend")

    def chat(self, messages: List[ChatMessage], **kwargs) -> Dict:
        if not self.api_key:
            raise ValueError(f"Missing API key for provider={self.provider_name}.")

        payload = {
            "model": kwargs.get("model", self.model_name),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if self.extra_body:
            payload.update(self.extra_body)

        cache_key = self._build_cache_key(payload)
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if isinstance(cached, dict) and "text" in cached:
                return cached

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_exc: Optional[Exception] = None
        total_attempts = self.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            start = time.time()
            try:
                proxies = {"http": None, "https": None, "all": None} if self.no_proxy else None
                response = requests.post(url, json=payload, headers=headers, timeout=self.timeout, proxies=proxies)
                latency_ms = int((time.time() - start) * 1000)

                if response.status_code == 429:
                    self._log_attempt(attempt, response.status_code, latency_ms, rate_limited=True)
                    self._sleep_backoff(attempt, total_attempts)
                    continue

                if response.status_code >= 500:
                    self._log_attempt(attempt, response.status_code, latency_ms)
                    self._sleep_backoff(attempt, total_attempts)
                    continue

                response.raise_for_status()
                data = response.json()

                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message", {})
                text = str(message.get("content", ""))

                result = {
                    "text": text,
                    "parsed_json": parse_json(text, default=None),
                    "raw": {
                        "id": data.get("id", ""),
                        "model": data.get("model", ""),
                        "usage": data.get("usage", {}),
                        "finish_reason": choice.get("finish_reason", ""),
                    },
                }

                if self.enable_response_logging:
                    self._log_success(attempt=attempt, latency_ms=latency_ms, text_len=len(text))

                if self.cache is not None:
                    self.cache.set(cache_key, result, ttl_seconds=self.cache_ttl_seconds)

                return result

            except requests.HTTPError as exc:
                latency_ms = int((time.time() - start) * 1000)
                last_exc = exc
                status = exc.response.status_code if exc.response is not None else "?"
                body = exc.response.text[:200] if exc.response is not None else ""
                self.logger.warning(
                    "llm_call provider=%s model=%s attempt=%d latency_ms=%d error=HTTPError status=%s body=%s",
                    self.provider_name, self.model_name, attempt, latency_ms, status, body,
                )
                self._sleep_backoff(attempt, total_attempts)
                continue
            except requests.RequestException as exc:
                latency_ms = int((time.time() - start) * 1000)
                last_exc = exc
                self._log_exception(attempt=attempt, latency_ms=latency_ms, err=exc)
                self._sleep_backoff(attempt, total_attempts)
                continue
            except ValueError as exc:
                # JSON decoding/serialization errors from requests response.
                last_exc = exc
                self._log_exception(attempt=attempt, latency_ms=int((time.time() - start) * 1000), err=exc)
                self._sleep_backoff(attempt, total_attempts)
                continue

        if last_exc is not None:
            raise RuntimeError(
                f"LLM request failed after {total_attempts} attempts for provider={self.provider_name}."
            ) from last_exc
        raise RuntimeError("LLM request failed with unknown error.")

    def _build_cache_key(self, payload: Dict) -> str:
        key_payload = {
            "provider": self.provider_name,
            "base_url": self.base_url,
            "payload": payload,
        }
        digest = hashlib.sha256(json.dumps(key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return f"llm::{digest}"

    def _sleep_backoff(self, attempt: int, total_attempts: int) -> None:
        if attempt >= total_attempts:
            return
        wait = min(self.retry_max_wait_sec, self.retry_backoff_sec * (2 ** (attempt - 1)))
        if wait > 0:
            time.sleep(wait)

    def _log_attempt(self, attempt: int, status_code: int, latency_ms: int, rate_limited: bool = False) -> None:
        if not self.enable_response_logging:
            return
        msg = "rate_limited" if rate_limited else "retryable_status"
        self.logger.warning(
            "llm_call provider=%s model=%s attempt=%d status=%d latency_ms=%d event=%s",
            self.provider_name,
            self.model_name,
            attempt,
            status_code,
            latency_ms,
            msg,
        )

    def _log_success(self, attempt: int, latency_ms: int, text_len: int) -> None:
        self.logger.info(
            "llm_call provider=%s model=%s attempt=%d status=200 latency_ms=%d text_len=%d",
            self.provider_name,
            self.model_name,
            attempt,
            latency_ms,
            text_len,
        )

    def _log_exception(self, attempt: int, latency_ms: int, err: Exception) -> None:
        if not self.enable_response_logging:
            return
        self.logger.warning(
            "llm_call provider=%s model=%s attempt=%d latency_ms=%d error=%s",
            self.provider_name,
            self.model_name,
            attempt,
            latency_ms,
            err.__class__.__name__,
        )
