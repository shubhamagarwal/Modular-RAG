from typing import Protocol


class EmbedderProtocol(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
