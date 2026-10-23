# 中国专利知识库助手

基于 LangGraph 的中国专利法知识库检索系统，支持条文定位、语义检索、多轮问答和带 PDF 页码的依据引用。

技术栈：LangGraph、LangChain、Chroma、SQLite、FastAPI、Streamlit。对话模型通过 Responses API 接入，向量模型通过兼容 OpenAI 的 Embeddings API 接入。

## 1. 环境准备

- Python 3.11 或 3.12，推荐 3.12。
- uv：安装并按 `uv.lock` 锁定 Python 依赖。
- Poppler：提供 PDF 文本提取命令 `pdftotext`。
- 支持 Responses、函数调用和 JSON 输出的对话 API，以及独立的向量 API。

进入仓库根目录后执行：

```bash
cd Chinese_patent_agent
# macOS 安装系统依赖
brew install uv poppler
uv sync --locked --python 3.12
```

Linux 可通过系统包管理器安装 Poppler（例如 Ubuntu 的 `poppler-utils`），另行安装 uv 后执行相同的依赖同步命令。以下命令均在仓库根目录执行。

## 2. 填写配置

首次安装时复制模板；已有 `.env` 时直接编辑，避免覆盖配置。

```bash
cp .env.example .env
chmod 600 .env
```

在 `.env` 中填写：

| 配置项 | 填写内容 |
|---|---|
| `CHAT_BASE_URL` | 对话平台基础地址，不包含 `/responses` |
| `CHAT_API_KEY` | 对话平台密钥 |
| `EMBEDDING_API_KEY` | 向量平台密钥 |

模板默认对话模型为 `gpt-6-astra`，向量模型为硅基流动 `Pro/BAAI/bge-m3`（1024 维）。使用其他平台或模型时，同时修改对应地址、模型名称及参数。完整说明见 [CONFIGURATION.md](CONFIGURATION.md)。

`.env`、原始资料、索引、会话、日志和生成报告均被 Git 忽略；公开模板中不要填写真实密钥。配置中的相对路径均以仓库根目录为基准。

## 3. 准备 PDF 资料

仓库不包含原始 PDF。创建资料目录，并按以下名称放置文件：

```bash
mkdir -p data/raw
```

| 资料 | 版本 | 保存位置 |
|---|---|---|
| 中华人民共和国专利法 | 2020 年修正 | `data/raw/patent_law_2020.pdf` |
| 中华人民共和国专利法实施细则 | 2023 年修订 | `data/raw/implementing_rules_2023.pdf` |
| 专利审查指南 | 2023 版 | `data/raw/examination_guidelines_2023.pdf` |

[资料清单](config/sources.manifest.json) 保存文件路径、SHA-256、页数和版本信息。解析器会校验文件哈希与页数；重命名不改变哈希，重新保存或替换 PDF 则可能改变哈希。当前解析规则适配清单中的三份 PDF，更换资料时需更新清单并验证章节、条号及页码解析结果，不能仅跳过哈希检查。

## 4. 检查依赖并构建知识库

```bash
# 本地依赖检查，不调用模型
.venv/bin/python -m patent_agent.cli doctor

# 可选：实际请求两个模型平台，产生 API 用量
.venv/bin/python -m patent_agent.cli doctor --live

# 从 PDF 提取并解析正文，不调用模型
.venv/bin/python -m patent_agent.cli parse --reextract

# 调用向量 API 建立索引，产生 API 用量
.venv/bin/python -m patent_agent.cli index
```

解析会保留 PDF 物理页码与条文层级，排除目录、页眉、页脚等重复内容。建索引按批次向量化，失败后可重新执行 `index` 继续构建；完整性检查通过后才激活新快照。

修改分块参数后重新执行 `parse` 和 `index`；更换向量模型或维度后重新执行 `index`。旧会话继续绑定原快照，新会话使用当前激活的快照。

## 5. 启动与停止

```bash
.venv/bin/python scripts/serve.py
```

该命令同时启动一个 API 进程和一个页面进程，按 `Ctrl-C` 停止两者。以后启动时，已有有效索引即可直接执行此命令。

| 服务 | 默认地址 |
|---|---|
| 问答页面 | <http://127.0.0.1:8501> |
| API 文档 | <http://127.0.0.1:8000/docs> |
| 知识库就绪检查 | <http://127.0.0.1:8000/health/ready> |

可在 `.env` 调整 `API_HOST`、`API_PORT`、`UI_HOST`、`UI_PORT`。修改 API 端口时同步修改页面使用的 `API_BASE_URL`，然后重启服务。

```bash
curl --fail http://127.0.0.1:8000/health/ready
curl --fail http://127.0.0.1:8501/_stcore/health
```

系统采用本地文件持久化、单后端 worker、单并发问答。默认仅监听本机，不包含用户认证；远程访问应通过 SSH 隧道或提供认证与 HTTPS 的反向代理。备份时停止服务并保存 `.env` 与完整 `data/` 目录，以保留资料、索引、快照及会话。

## 常见启动问题

| 问题 | 处理方式 |
|---|---|
| 配置校验失败 | 检查 `.env` 的必填项及 API 基础地址格式 |
| 找不到 `pdftotext` | 安装 Poppler，确认命令在 PATH 中 |
| PDF 哈希或页数不一致 | 核对文件与清单，替换资料时验证解析规则 |
| 就绪检查失败或没有可用快照 | 依次执行 `parse --reextract`、`index`，检查命令结果 |
| API 返回 401 / 429 | 核对对应平台密钥、额度和限流设置 |
| 端口被占用 | 停止占用进程或修改端口，保持 `API_BASE_URL` 一致 |

诊断、解析和索引报告自动写入 `reports/`，运行日志写入 `data/logs/`。就绪检查只读取本地状态，不请求模型。

## 命令行与测试

```bash
.venv/bin/python -m patent_agent.cli ask '专利法第二十二条规定了什么？'
# 将 SESSION_ID 替换为上次返回的 session_id
.venv/bin/python -m patent_agent.cli ask '它对实用新型的要求呢？' --session SESSION_ID

.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
```

语料相关测试需要先准备上述 PDF 并构建索引。以下可选评测会调用真实模型，产生 API 用量：

```bash
.venv/bin/python scripts/evaluate.py
.venv/bin/python scripts/summarize_evaluation.py
```

题集位于 `eval/cases.json`；结果写入 `reports/evaluation.jsonl`，按快照和题目 ID 断点续跑。完整重跑前应归档旧结果。汇总工具按快照分别统计引用完整性、检索预算和预设行为检查，不将这些指标作为法律语义准确率。

## 功能与目录

模型通过 LangGraph 选择精确定位或语义检索工具，检查证据，必要时补查，生成回答并校验引用。默认最多两轮检索、每轮两个工具、一次引用修复。页面提供证据、完整条文、PDF 对应页及执行过程查看。

```text
Chinese_patent_agent/
├── .env.example         # 配置模板
├── config/              # 资料路径、哈希与版本清单
├── src/patent_agent/    # 解析、索引、检索、工作流、API 和页面
├── scripts/             # 启动与评测命令
├── tests/               # 自动化测试
├── eval/                # 评测题集
├── data/                # 本地资料、索引、会话及日志，运行时生成
├── reports/             # 运行报告，按需生成
├── pyproject.toml       # Python 依赖和工具配置
└── uv.lock              # 依赖锁定文件
```

资料版本以清单为准，系统不自动更新法规。PDF 文本解析不完整覆盖图片、化学结构、复杂公式和表格。引用校验检查来源、快照与逐字引文，回答解释是否充分受原文支持仍需结合证据判断。
