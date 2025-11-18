import asyncio
from typing import List, Dict, Optional

from google import genai
from google.genai import types

from config_loader import configToml


class _ChatResponse:
    def __init__(self, content: str):
        self.content = content


class GeminiAPIHandler:
    """A lightweight adapter to mimic Ollama chat interface using Gemini GenAI.

    Expected input: List[{"role": str, "content": str}]
    Returns: object with `.content` string, similar to previous API.
    """

    def __init__(self):
        http_options = {
            "base_url": configToml["llmChat"]["link_build"],
        }
        self._model = configToml["llmChat"]["modelChat"]
        self._client = genai.Client(
            api_key=configToml["apiToken"]["gemini_llm"][0],
            http_options=http_options,
        )

    async def chat(self, messages: List[Dict[str, str]]) -> _ChatResponse:
        # Extract optional system instructions and build contents
        system_segments: List[str] = []
        contents: List[types.Content] = []

        for m in messages:
            role = m.get("role", "user")
            text = m.get("content", "")
            if not isinstance(text, str):
                text = str(text)

            if role == "system":
                system_segments.append(text)
                continue

            mapped_role = "model" if role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=mapped_role,
                    parts=[types.Part.from_text(text)],
                )
            )

        # Fallback: prepend system text as a user-prefixed instruction for reliability
        if system_segments:
            sys_text = "\n".join(system_segments)
            contents.insert(
                0,
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(sys_text)],
                ),
            )

        try:
            resp = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
            )
            text = resp.text or ""
        except Exception as e:
            text = f"GenAI Error: {e}"

        return _ChatResponse(text)

    async def close(self):
        # Current genai client doesn't require explicit close; keep API compatibility.
        await asyncio.sleep(0)
