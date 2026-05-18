from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Indo Legal AI"
    app_version: str = "0.1.0"

    database_url: str = "postgresql+psycopg2://legalai:legalai@db:5432/legalai"

    # Embedding model + vector dimension. Keep these in lockstep.
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = 768

    # Cross-encoder reranker. Disable if the host is resource-constrained.
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_enabled: bool = True

    # Retrieval sizing.
    retrieval_top_k: int = 5
    rerank_candidates: int = 20

    # LLM provider: "ollama" or "openai".
    llm_provider: str = "ollama"

    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "llama3.2:latest"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"

    llm_timeout_seconds: float = 120.0
    conversation_memory_turns: int = 4

    # A grounded citation counts as "verified" when this fraction of the cited
    # Pasal's content words also appear in the answer.
    citation_support_threshold: float = 0.4

    # A sentence of the answer counts as "grounded" when this fraction of its
    # content words appear in some retrieved context chunk.
    claim_grounding_threshold: float = 0.5


settings = Settings()
