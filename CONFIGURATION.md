# 本地配置与外部依赖

配置文件仅使用项目根目录 `.env`。建议通过 `chmod 600 .env` 限制文件权限；该文件已被 Git 忽略。`.env.example` 是可以公开的中文填写模板；不要将真实密钥填进模板。

| 依赖 | 是否必须 | 当前用途 |
|---|---|---|
| 对话平台 Base URL / API Key | 必须 | `gpt-6-astra` 的 Responses、函数调用和 JSON 输出 |
| 硅基流动 API Key | 必须 | `Pro/BAAI/bge-m3` 文本向量化 |
| Python、uv | 必须 | 本地环境和依赖锁定；推荐 Python 3.12 |
| Poppler / pdftotext | 重新提取 PDF 时需要 | 保留版式的逐页文字提取 |
| Chroma、SQLite | 必须，无外部账号 | 都运行在本地；随 Python 依赖安装 |
| LangSmith | 不需要 | 当前适配器明确关闭云端追踪，未实现该可选集成 |
| 云服务器、Redis、Docker、额外数据库账号 | 不需要 | 无需配置 |

## 必填位置

```dotenv
# 填写平台基础地址，例如 https://你的平台/v1；不包含 /responses。
CHAT_BASE_URL=
# 填写对话平台分配的 API Key。
CHAT_API_KEY=
# 填写硅基流动控制台生成的 API Key。
EMBEDDING_API_KEY=
```

模板默认配置：

```dotenv
CHAT_API_MODE=responses
CHAT_MODEL=gpt-6-astra
CHAT_REASONING_EFFORT=low
CHAT_SEND_TEMPERATURE=false
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=Pro/BAAI/bge-m3
EMBEDDING_DIMENSIONS=1024
EMBEDDING_SEND_DIMENSIONS=false
EMBEDDING_ENCODING_FORMAT=float
LANGSMITH_TRACING=false
```

`EMBEDDING_DIMENSIONS` 用于校验输出维度。BGE-M3 请求不携带 `dimensions`；设置 `EMBEDDING_SEND_DIMENSIONS=true` 会被配置检查阻止。

## 可调整项及影响

- `EMBEDDING_BATCH_SIZE`：建库批大小，默认 10；影响请求数量与失败后的重做范围。
- `CHUNK_TARGET_CHARS / CHUNK_MAX_CHARS / CHUNK_OVERLAP_CHARS`：默认 700 / 1200 / 100。修改后重新 parse 和 index，生成新快照。
- `RETRIEVAL_TOP_K / RETRIEVAL_PER_SOURCE_K`：默认总召回 8、每来源 4；检索会优先保留所请求来源的最佳结果，再按距离补足。
- `CONTEXT_MAX_CHUNKS / CONTEXT_MAX_CHARS`：默认最终 6 个完整片段、最多 10,000 字符；实际模型输入还包含指令、工具协议和少量历史消息。
- `MAX_RETRIEVAL_ROUNDS / MAX_TOOLS_PER_ROUND / MAX_CITATION_REPAIRS`：默认 2 / 2 / 1，配置上限分别为 2 / 2 / 1。
- `CHAT_HISTORY_TURNS`：默认向模型提供最近 4 轮已完成问答；历史内容不自动作为证据。
- `MODEL_TIMEOUT_SECONDS / MODEL_MAX_RETRIES`：默认每次请求 45 秒、最多重试 2 次。各适配器只有一层重试；认证错误不进行向量请求重试。
- `RUN_TIMEOUT_SECONDS`：默认 180 秒，节点和请求预算共同控制；并非进程级强制中止。
- `APP_MAX_CONCURRENT_RUNS`：只接受 1。后端也只使用 1 个 worker。
- `API_HOST / API_PORT / UI_HOST / UI_PORT / API_BASE_URL`：默认本机 8000 / 8501，页面通过 `API_BASE_URL` 请求后端。
- 数据路径相对于项目根目录解析；`SOURCES_MANIFEST_PATH` 指向资料清单，`RAW_DATA_DIR` 约束原 PDF 服务目录，`PROCESSED_DATA_DIR` 保存正文和片段，SQLite、Chroma 与日志均在 `data/` 下。
- `LOG_LEVEL` 控制本地元数据日志；`LOG_FULL_MODEL_PAYLOADS` 必须为 false。云端 tracing 相关字段保留在模板中，但当前只支持关闭状态。

`APP_ENV` 标记运行环境，不代表已经实现不同环境的部署策略。配置项及其默认值见 `.env.example` 和 `src/patent_agent/settings.py`。

## 诊断顺序

1. `.venv/bin/python -m patent_agent.cli doctor`：检查本地图、SQLite 和向量库。
2. `.venv/bin/python -m patent_agent.cli doctor --live`：付费检查真实 API，报告 `HTTP 401` 时核对对应密钥，`429` 时检查限流/额度；不把模型平台的响应正文输出到终端。
3. 查看 `reports/doctor-live.json`、`reports/parsing.json` 和 `reports/index.json`。
4. `/health/ready` 只检查本地快照是否可用，不会因为打开页面就调用模型。

修改 `.env` 后重启服务。更换向量模型或维度后必须重新建索引；不要用不同模型的查询向量检索旧 collection。
