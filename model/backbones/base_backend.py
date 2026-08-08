from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ChatMessage:
    role: str
    content: str


class BaseLLMBackend(ABC):
    """Abstract backend for LLM providers."""

    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 1024):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def chat(self, messages: List[ChatMessage], **kwargs) -> Dict:
        raise NotImplementedError

    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict:
        messages: List[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))
        return self.chat(messages, **kwargs)
