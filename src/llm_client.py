"""Thin wrapper around the OpenAI-compatible chat-completion API."""

from __future__ import annotations

from typing import List

from openai import OpenAI


class LLMClient:
    """Wrapper around an OpenAI-compatible chat-completion endpoint.

    Parameters
    ----------
    api_key:
        API key for the LLM service.
    base_url:
        Base URL for the API endpoint.  Default is OpenAI's public endpoint.
    model:
        Model identifier string (e.g. "gpt-4o").
    temperature:
        Sampling temperature passed to the model.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        temperature: float = 0.7,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def chat(
        self,
        messages: List[dict],
        temperature: float | None = None,
    ) -> str:
        """Send a list of chat messages and return the assistant reply as a string.

        Parameters
        ----------
        messages:
            List of ``{"role": ..., "content": ...}`` dicts.
        temperature:
            Override the default temperature for this call.

        Returns
        -------
        str
            The content of the assistant message.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(
                "LLM returned an empty (None) message content. "
                "Check your model settings or API quota."
            )
        return content
