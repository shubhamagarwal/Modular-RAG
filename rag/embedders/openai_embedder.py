from openai import AsyncOpenAI

from rag.config import settings
from rag.logger import get_logger

log = get_logger("embedders.openai")


class OpenAIEmbedder:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.github_token,
            base_url=settings.api_base_url,
        )
        self._model = settings.openai_embedding_model
        self._batch_size = 100

    async def embed(self, texts: list[str]) -> list[list[float]]:
        total = len(texts)
        num_batches = (total + self._batch_size - 1) // self._batch_size
        log.info("Embedding %d texts in %d batch(es) using %s", total, num_batches, self._model)
        results: list[list[float]] = []
        for i in range(0, total, self._batch_size):
            batch = texts[i: i + self._batch_size]
            batch_num = i // self._batch_size + 1
            log.debug("  Batch %d/%d — %d texts", batch_num, num_batches, len(batch))
            response = await self._client.embeddings.create(model=self._model, input=batch)
            results.extend(item.embedding for item in response.data)
        log.info("Embedding complete — %d vectors returned (dim=%d)", len(results), len(results[0]) if results else 0)
        return results
