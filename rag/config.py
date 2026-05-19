from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    github_token: str
    api_base_url: str = "https://models.inference.ai.azure.com"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    chroma_persist_dir: str = ".chroma"
    chroma_collection: str = "codebase"

    chunk_size: int = 1200
    chunk_overlap: int = 200

    retrieval_top_k: int = 10
    rerank_top_k: int = 4
    context_max_tokens: int = 6000


settings = Settings()
