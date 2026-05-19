from typing import AsyncIterator

from openai import AsyncOpenAI

from rag.config import settings
from rag.logger import get_logger

log = get_logger("llm.openai")

_SYSTEM_PROMPT = (
    "You are a codebase assistant. Answer questions using only the provided code snippets. "
    "Always cite the file path and line numbers when referencing code. "
    "If the answer cannot be found in the snippets, say so clearly."
)


class OpenAILLM:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.github_token,
            base_url=settings.api_base_url,
        )
        self._model = settings.openai_chat_model

    async def generate(
        self,
        system: str,
        user: str,
        context: str,
    ) -> AsyncIterator[str]:
        full_user = f"Code context:\n\n{context}\n\n---\n\nQuestion: {user}" if context else user
        log.info("Sending request to %s (model=%s, context_len=%d chars)", settings.api_base_url, self._model, len(context))
        stream = await self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            stream=True,
            messages=[
                {"role": "system", "content": system or _SYSTEM_PROMPT},
                {"role": "user", "content": full_user},
            ],
        )
        log.info("Streaming response from LLM...")
        token_count = 0
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                token_count += 1
                yield delta
        log.info("LLM stream complete — %d tokens received", token_count)
