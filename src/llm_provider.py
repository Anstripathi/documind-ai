"""LLM provider interface for cloud and local model backends."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMError(Exception):
    """Base class for provider failures safe to show to users."""


class LLMAuthError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMUnavailableError(LLMError):
    pass


class LLMProvider(ABC):
    name: str = "llm"

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        ...


class GeminiProvider(LLMProvider):
    """Google Gemini via the google-genai SDK."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        if not api_key:
            raise LLMAuthError("No Gemini API key configured.")
        self.model = model
        self.name = f"gemini:{model}"
        self._api_key = api_key

    def generate(self, system: str, user: str) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMUnavailableError(
                "google-genai package is not installed."
            ) from exc

        try:
            client = genai.Client(api_key=self._api_key)
            resp = client.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.0,
                    max_output_tokens=1024,
                ),
            )
            text = (resp.text or "").strip()
        except Exception as exc:
            code = getattr(exc, "code", None)
            msg = str(exc).lower()

            if code in (401, 403) or "apikey" in msg or "permission" in msg:
                raise LLMAuthError(
                    "The LLM API key is invalid or not permitted."
                ) from exc

            if code == 429 or "quota" in msg or "rate" in msg:
                raise LLMRateLimitError(
                    "LLM rate limit reached. Please retry shortly."
                ) from exc

            raise LLMUnavailableError(
                "The Gemini provider is currently unavailable."
            ) from exc

        if not text:
            raise LLMUnavailableError(
                "The LLM returned an empty response."
            )

        return text


class OllamaProvider(LLMProvider):
    """Local Ollama provider using its HTTP chat API."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        if not model.strip():
            raise ValueError("Ollama model name cannot be empty.")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }

        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))

        except HTTPError as exc:
            if exc.code == 404:
                raise LLMUnavailableError(
                    f"Ollama model '{self.model}' was not found. "
                    "Check the installed model name."
                ) from exc

            if exc.code == 429:
                raise LLMRateLimitError(
                    "Ollama is busy. Please retry shortly."
                ) from exc

            raise LLMUnavailableError(
                f"Ollama returned HTTP error {exc.code}."
            ) from exc

        except (URLError, TimeoutError, OSError) as exc:
            raise LLMUnavailableError(
                "Cannot connect to Ollama. Check that Ollama is running."
            ) from exc

        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LLMUnavailableError(
                "Ollama returned an invalid response."
            ) from exc

        text = (
            result.get("message", {}).get("content", "")
            if isinstance(result, dict)
            else ""
        )

        if not isinstance(text, str) or not text.strip():
            raise LLMUnavailableError(
                "Ollama returned an empty response."
            )

        return text.strip()


def build_provider(
    api_key: str | None,
    model: str,
    provider: str = "gemini",
    base_url: str = "http://localhost:11434",
    timeout: int = 120,
) -> LLMProvider | None:
    """Build the selected provider, or None for extractive-only mode."""

    provider = provider.strip().lower()

    if provider == "ollama":
        return OllamaProvider(
            model=model,
            base_url=base_url,
            timeout=timeout,
        )

    if provider == "gemini":
        if not api_key:
            return None
        return GeminiProvider(api_key=api_key, model=model)

    if provider == "none":
        return None

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )
