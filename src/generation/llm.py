"""Minimal client for a local Ollama server (REST API, no SDK).

Only one call is needed for RAG: send chat messages, get the reply text.
Writing it with plain HTTP shows exactly what goes over the wire, and it is
easy to replace with a fake in tests.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


class LLMError(RuntimeError):
    """The LLM call failed (bad model name, server error, unexpected reply)."""


class LLMUnavailableError(LLMError):
    """Ollama could not be reached or did not answer in time."""


class OllamaClient:
    """Sends chat messages to Ollama's /api/chat endpoint and returns the reply."""

    def __init__(
        self,
        url: str,
        model: str,
        temperature: float = 0.0,
        num_ctx: int = 4096,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """
        Args:
            url: base URL, e.g. "http://localhost:11434".
            model: Ollama model tag, e.g. "llama3.2:3b".
            temperature: 0 gives the same answer every run (repeatable evaluation).
            num_ctx: context window in tokens. Ollama's default is small, and longer
                prompts are cut from the START, where the system rules are.
            timeout: seconds to wait. The first call also loads the model into RAM.
            transport: only for tests (httpx.MockTransport), to fake the server.
        """
        self.model = model
        self.options = {"temperature": temperature, "num_ctx": num_ctx}
        self._url = url.rstrip("/")
        # One Client reuses the TCP connection across calls (faster than httpx.post each time).
        self._client = httpx.Client(base_url=self._url, timeout=timeout, transport=transport)

    def chat(self, messages: list[Message]) -> str:
        """Return the model's reply to `messages`.

        Raises:
            LLMUnavailableError: server not running, unreachable or too slow.
            LLMError: model not pulled, server error, or a reply in an unexpected format.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,  # one JSON reply instead of a token stream
            "options": self.options,
        }

        start = time.perf_counter()
        try:
            response = self._client.post("/api/chat", json=payload)
        except httpx.ConnectError as e:
            raise LLMUnavailableError(
                f"Cannot reach Ollama at {self._url}. Is it running? (start the Ollama app or `ollama serve`)"
            ) from e
        except httpx.TimeoutException as e:
            raise LLMUnavailableError(
                f"Ollama did not answer within {self._client.timeout.read}s (model {self.model})."
            ) from e

        if response.status_code == 404:
            raise LLMError(f"Model {self.model!r} not found in Ollama. Run: ollama pull {self.model}")
        if response.status_code != 200:
            raise LLMError(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")

        try:
            data = response.json()
            content: str = data["message"]["content"]
        except (ValueError, KeyError, TypeError) as e:
            raise LLMError(f"Unexpected reply from Ollama: {response.text[:200]}") from e

        logger.info(
            "LLM %s: %.2fs, prompt %s tokens, answer %s tokens",
            self.model,
            time.perf_counter() - start,
            data.get("prompt_eval_count", "?"),
            data.get("eval_count", "?"),
        )
        # If the prompt hit num_ctx, Ollama truncated it silently. Warn so it never goes unnoticed.
        if data.get("prompt_eval_count", 0) >= self.options["num_ctx"]:
            logger.warning(
                "Prompt filled the whole context window (%d tokens): it was probably truncated",
                self.options["num_ctx"],
            )
        return content.strip()

    def close(self) -> None:
        self._client.close()
