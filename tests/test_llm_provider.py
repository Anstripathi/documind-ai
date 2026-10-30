import io
import json
from urllib.error import HTTPError, URLError
from unittest.mock import patch

import pytest

from src.llm_provider import (
    LLMUnavailableError,
    OllamaProvider,
    build_provider,
)


def test_build_ollama_provider_without_api_key():
    provider = build_provider(
        None,
        "qwen3:4b-instruct-2507-q4_K_M",
        provider="ollama",
    )
    assert isinstance(provider, OllamaProvider)
    assert provider.name == "ollama:qwen3:4b-instruct-2507-q4_K_M"


def test_build_gemini_without_key_returns_none():
    assert build_provider(None, "gemini-2.5-flash") is None


@patch("src.llm_provider.urlopen")
def test_ollama_generate_sends_system_and_user_messages(mock_urlopen):
    response_data = {
        "message": {
            "content": "The answer is supported by the document."
        }
    }
    mock_urlopen.return_value = io.BytesIO(
        json.dumps(response_data).encode("utf-8")
    )

    provider = OllamaProvider(
        "qwen3:4b-instruct-2507-q4_K_M"
    )

    result = provider.generate("System instructions", "User question")

    assert result == "The answer is supported by the document."

    request = mock_urlopen.call_args.args[0]
    payload = json.loads(request.data.decode("utf-8"))

    assert request.full_url == "http://localhost:11434/api/chat"
    assert payload["model"] == "qwen3:4b-instruct-2507-q4_K_M"
    assert payload["stream"] is False
    assert payload["messages"] == [
        {"role": "system", "content": "System instructions"},
        {"role": "user", "content": "User question"},
    ]


@patch("src.llm_provider.urlopen")
def test_ollama_connection_error_is_mapped(mock_urlopen):
    mock_urlopen.side_effect = URLError("connection refused")

    provider = OllamaProvider("qwen3:4b")

    with pytest.raises(LLMUnavailableError, match="Cannot connect"):
        provider.generate("system", "question")


@patch("src.llm_provider.urlopen")
def test_ollama_missing_model_is_reported(mock_urlopen):
    mock_urlopen.side_effect = HTTPError(
        "http://localhost:11434/api/chat",
        404,
        "Not Found",
        {},
        None,
    )

    provider = OllamaProvider("missing-model")

    with pytest.raises(LLMUnavailableError, match="was not found"):
        provider.generate("system", "question")


@patch("src.llm_provider.urlopen")
def test_ollama_empty_response_is_rejected(mock_urlopen):
    mock_urlopen.return_value = io.BytesIO(
        json.dumps({"message": {"content": ""}}).encode("utf-8")
    )

    provider = OllamaProvider("qwen3:4b")

    with pytest.raises(LLMUnavailableError, match="empty response"):
        provider.generate("system", "question")
