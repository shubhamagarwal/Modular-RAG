from typing import AsyncIterator, Protocol


class LLMProtocol(Protocol):
    def generate(
        self,
        system: str,
        user: str,
        context: str,
    ) -> AsyncIterator[str]: ...
