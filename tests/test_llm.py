"""Tests for the Ollama client, using httpx.MockTransport as a fake server (no Ollama needed)."""

from __future__ import annotations

import json

import httpx
import pytest

from generation.llm import LLMError, LLMUnavailableError, OllamaClient

MESSAGES = [{"role": "user", "content": "hi"}]


def client_with(handler) -> OllamaClient:
    return OllamaClient(
        "http://ollama:11434/",
        "llama3.2:3b",
        temperature=0.0,
        num_ctx=4096,
        transport=httpx.MockTransport(handler),
    )


def test_sends_expected_request_and_returns_stripped_content():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "  Hello.  "}})

    assert client_with(handler).chat(MESSAGES) == "Hello."
    assert seen["url"] == "http://ollama:11434/api/chat"  # trailing slash in base URL handled
    body = seen["body"]
    assert body["model"] == "llama3.2:3b"
    assert body["messages"] == MESSAGES
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "num_ctx": 4096}


def test_server_down_raises_unavailable():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(LLMUnavailableError, match="Is it running"):
        client_with(handler).chat(MESSAGES)


def test_timeout_raises_unavailable():
    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(LLMUnavailableError, match="did not answer"):
        client_with(handler).chat(MESSAGES)


def test_unavailable_is_an_llm_error():
    # Callers can catch LLMError alone and still handle "server down".
    assert issubclass(LLMUnavailableError, LLMError)


def test_missing_model_gives_pull_hint():
    handler = lambda request: httpx.Response(404, json={"error": "model not found"})  # noqa: E731
    with pytest.raises(LLMError, match="ollama pull llama3.2:3b"):
        client_with(handler).chat(MESSAGES)


def test_server_error_raises():
    handler = lambda request: httpx.Response(500, text="out of memory")  # noqa: E731
    with pytest.raises(LLMError, match="500.*out of memory"):
        client_with(handler).chat(MESSAGES)


@pytest.mark.parametrize("reply", ["not json", json.dumps({"no_message": 1})])
def test_malformed_reply_raises(reply):
    handler = lambda request: httpx.Response(200, text=reply)  # noqa: E731
    with pytest.raises(LLMError, match="Unexpected reply"):
        client_with(handler).chat(MESSAGES)


def test_full_context_window_logs_truncation_warning(caplog):
    handler = lambda request: httpx.Response(  # noqa: E731
        200, json={"message": {"content": "ok"}, "prompt_eval_count": 4096}
    )
    client_with(handler).chat(MESSAGES)
    assert "truncated" in caplog.text
