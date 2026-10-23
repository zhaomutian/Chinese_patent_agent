from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="forbid", hide_input_in_errors=True)
    chat_provider: str = "openai_compatible"
    chat_api_mode: str = "responses"
    chat_reasoning_effort: str = "low"
    chat_send_temperature: bool = False
    chat_base_url: SecretStr
    chat_api_key: SecretStr
    chat_model: str = "gpt-6-astra"
    embedding_provider: str = "openai_compatible"
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: SecretStr
    embedding_model: str = "Pro/BAAI/bge-m3"
    embedding_dimensions: int = Field(default=1024, gt=0)
    embedding_send_dimensions: bool = False
    embedding_batch_size: int = Field(default=10, ge=1, le=100)
    embedding_encoding_format: str = "float"
    app_env: str = "local"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    ui_host: str = "127.0.0.1"
    ui_port: int = 8501
    api_base_url: str = "http://127.0.0.1:8000"
    sources_manifest_path: Path = Path("config/sources.manifest.json")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    chroma_persist_dir: Path = Path("data/chroma")
    chroma_collection_prefix: str = "patent_kb"
    catalog_db_path: Path = Path("data/sqlite/catalog.db")
    checkpoint_db_path: Path = Path("data/sqlite/checkpoints.db")
    log_dir: Path = Path("data/logs")
    log_level: str = "INFO"
    log_full_model_payloads: bool = False
    chat_temperature: float = 0
    chat_max_output_tokens: int = 2500
    model_timeout_seconds: float = Field(default=45, gt=0)
    model_max_retries: int = Field(default=2, ge=0, le=5)
    chunk_target_chars: int = 700
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 100
    retrieval_top_k: int = 8
    retrieval_per_source_k: int = 4
    context_max_chunks: int = 6
    context_max_chars: int = 10000
    max_retrieval_rounds: int = Field(default=2, ge=1, le=2)
    max_tools_per_round: int = Field(default=2, ge=1, le=2)
    max_citation_repairs: int = Field(default=1, ge=0, le=1)
    graph_recursion_limit: int = 30
    run_timeout_seconds: float = 180
    app_max_concurrent_runs: int = 1
    chat_history_turns: int = 4
    question_max_chars: int = 2000
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_project: str = "patent-law-agent"
    langsmith_endpoint: str = ""

    @model_validator(mode="after")
    def validate_config(self):
        from urllib.parse import urlparse

        for url in (self.chat_base_url.get_secret_value(), self.embedding_base_url):
            p = urlparse(url)
            if p.scheme != "https" or not p.netloc or p.username or p.password or p.query:
                raise ValueError("模型基础地址须为不带认证信息和查询串的 HTTPS 地址")
            if p.path.rstrip("/").endswith(("/responses", "/embeddings", "/chat/completions")):
                raise ValueError("请填写基础地址，不包含接口后缀")
        if not self.chat_api_key.get_secret_value() or not self.embedding_api_key.get_secret_value():
            raise ValueError("请在 .env 填写两项 API 密钥")
        if self.chat_provider != "openai_compatible" or self.embedding_provider != "openai_compatible":
            raise ValueError("当前应用仅支持 OpenAI compatible API")
        if self.embedding_encoding_format != "float":
            raise ValueError("当前向量适配器需要 float 编码")
        if self.chat_model == "gpt-6-astra" and self.chat_send_temperature:
            raise ValueError("GPT-6 Astra 不发送 temperature 参数")
        if self.chat_api_mode != "responses" or self.chat_reasoning_effort == "none":
            raise ValueError("当前适配器需要 Responses 和非 none 的推理强度")
        if "bge-m3" in self.embedding_model.lower() and self.embedding_send_dimensions:
            raise ValueError("BGE-M3 不发送 dimensions 参数")
        if not 0 <= self.chunk_overlap_chars < self.chunk_target_chars <= self.chunk_max_chars:
            raise ValueError("分块长度配置不合法")
        if self.log_full_model_payloads or self.langsmith_tracing:
            raise ValueError("当前应用仅实现本地脱敏日志；完整载荷和云端 tracing 应保持关闭")
        if self.app_max_concurrent_runs != 1:
            raise ValueError("当前应用仅支持单进程单并发")
        for name in (
            "sources_manifest_path",
            "raw_data_dir",
            "processed_data_dir",
            "chroma_persist_dir",
            "catalog_db_path",
            "checkpoint_db_path",
            "log_dir",
        ):
            value = getattr(self, name)
            setattr(self, name, value if value.is_absolute() else ROOT / value)
        return self
