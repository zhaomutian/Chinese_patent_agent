# 中国专利知识库助手

基于 LangGraph 的中国专利知识库助手，覆盖国内专利法规与国际条约，支持条文定位、语义检索、多轮问答、题目分析、法律解释及带 PDF 页码的依据引用。

技术栈：LangGraph、LangChain、Chroma、SQLite、FastAPI、Streamlit。对话模型通过 Responses API 接入，向量模型通过兼容 OpenAI 的 Embeddings API 接入。

文档入口：[配置与外部依赖](CONFIGURATION.md) · [题目分析与解释审核](docs/EXAM_ANALYSIS_DESIGN.md) · [国际条约范围与导入](docs/INTERNATIONAL_SOURCES.md)。首次部署依次完成下方第 1–5 步；已有环境和索引时直接执行启动命令。

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

[资料清单](config/sources.manifest.json) 保存文件路径、SHA-256、页数和版本信息。解析器会校验文件哈希与页数；重命名不改变哈希，重新保存或替换 PDF 则可能改变哈希。国内资料解析规则适配上述三份 PDF。清单还登记了 15 份来自 WIPO/WTO 的条约及配套文本，可用以下命令恢复官方原件：

```bash
.venv/bin/python scripts/download_treaties.py
```

下载器校验固定哈希，遇到官方文件变化时停止，避免未经核对替换版本。条约范围、版本差异、条款定位及附件限制见[国际资料说明](docs/INTERNATIONAL_SOURCES.md)。更换资料时需更新清单并验证章节、条号及页码解析结果，不能仅跳过哈希检查。

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

解析会保留 PDF 物理页码与条文层级，排除目录、页眉、页脚等重复内容。建索引按批次向量化，原文会发送到 `.env` 中配置的向量平台并产生用量；片段 ID 与向量配置一致时复用历史快照向量。失败后可重新执行 `index` 继续构建；完整性检查通过后才激活新快照。

修改分块参数后重新执行 `parse` 和 `index`；更换向量模型或维度后重新执行 `index`。旧会话继续绑定原快照，新会话使用当前激活的快照。导入国际条约后，在页面点击“开始新会话”才能使用新资料；重启服务不会改变旧会话绑定的快照。

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

## 题目分析与讲解

在页面选择“题目分析”，粘贴完整题干、选项和单选／多选要求。系统提取事实与争点、选择分析方法、检索规则及例外，区分原文直接支持与一般规则的解释适用。解释性判断经独立审核后，再按设问方向汇总答案。同一会话可追问“为什么 B 不对”、修改日期或条件；修改题设后重新检索分析。选择“只给提示”时仅提供审题检查步骤，不生成解答。

此功能复用已有对话、向量 API 和法规索引，无需新账号。方法卡提供分析程序，法律结论仍须引用法规原文。一般规定没有逐字列举程序名称时，可以说明适用关系；不能据此扩大主体或直接推出未规定的处罚。页面可展开“原文规定与本题适用”。运行时不读取开发题的教材答案；证据不足或题干有歧义时会说明缺口或追问。教材答案是离线评分的正确标准。

### 导入开发材料和生成方法卡

准备自己有权使用的开发集 JSONL，每行包含 `question_id`、`chapter_id`、`chapter_name`、`stem`、`options`（选项标签到文本的映射）、`reference_answer`、`explanation`、`split: "development"`。可保留其他元数据。开发集、验证集和测试集应事先按相似题组划分；本命令只接受开发集。

```bash
# 替换路径为自己的开发集；仅在本地检查并生成逐章请求预览
.venv/bin/python scripts/import_teaching.py --source path/to/development.jsonl --preview

# 检查 data/teaching/request-preview/ 后提炼方法卡
# 会将开发题干、选项、教材答案及解析发送至 CHAT_BASE_URL，产生 API 用量
.venv/bin/python scripts/import_teaching.py --source path/to/development.jsonl --build-methods
```

方法卡保存在 `data/teaching/methods/`，全部章节完成后原子切换有效版本；失败可重复运行，复用已完成章节。一个会话在首次分析时绑定方法版本，之后更新方法包需新建会话使用。新机器可复制整个方法目录；未配置方法包时仍可使用基础审题与法规检索，但不会自动获得教材方法。

```bash
.venv/bin/python -m patent_agent.cli ask '粘贴完整题干和选项' --mode analyze --explanation detailed
.venv/bin/python -m patent_agent.cli ask '为什么 B 不对？' --session SESSION_ID --mode analyze
.venv/bin/python -m patent_agent.cli ask '只给我提示' --session SESSION_ID --explanation hint

# 开发样例回归：参考答案仅进入评分器，不进入解题请求；实际调用模型
.venv/bin/python scripts/evaluate_exam.py --questions data/teaching/development.jsonl --dataset development --limit 5 --output reports/exam-development.jsonl
```

通用解释方法内置可用，也可从开发材料提炼：

```bash
# 默认仅生成本地预览，三类方法每类最多使用8道开发题
.venv/bin/python scripts/build_interpretation_methods.py --source data/teaching/development.jsonl
# 已确认材料可发送到对话平台后执行；会产生API费用
.venv/bin/python scripts/build_interpretation_methods.py --source data/teaching/development.jsonl --build
# 真实模型正反例：明确标注为自拟，不计入教材准确率
.venv/bin/python scripts/evaluate_interpretation.py
```

解释方法保存在 `data/teaching/methods/interpretation/`，失败不切换有效包，更新后新建会话使用。章节卡最多两张、解释卡最多一张，共用方法字符预算。API 的 `exam_analysis` 增加 `answer_complete`、`explanation_complete`、`review_status` 与各选项的规则原文和适用摘要。额外讲解未完成时，可以保留已核实的答案并说明缺口。

独立评测环境可使用 `--questions` 指定只含题干和选项的文件，通过 `--references` 指定教材评分文件，使用 `--dataset validation` 或 `--dataset test`。不得把保留评测材料交给方法提炼流程。评测采用完整选项集合匹配；未完成、需要澄清的题目同样计为未命中，并单独记录状态。开发集结果不能代表保留集准确率。

API 的 `POST /api/v1/chat` 增加可选字段 `mode`（`auto`／`lookup`／`analyze`）及 `explanation_level`（`hint`／`concise`／`detailed`）。响应增加逐项判断、争点覆盖、题目版本及方法版本，仍兼容原有回答与引用字段。当前只支持文本输入，详细结构与限制见 [题目分析实现说明](docs/EXAM_ANALYSIS_DESIGN.md)。

## 常见启动问题

| 问题 | 处理方式 |
|---|---|
| 配置校验失败 | 检查 `.env` 的必填项及 API 基础地址格式 |
| 找不到 `pdftotext` | 安装 Poppler，确认命令在 PATH 中 |
| PDF 哈希或页数不一致 | 核对文件与清单，替换资料时验证解析规则 |
| 就绪检查失败或没有可用快照 | 依次执行 `parse --reextract`、`index`，检查命令结果 |
| API 返回 401 / 429 | 核对对应平台密钥、额度和限流设置 |
| 导入条约后仍提示缺少资料 | 点击“开始新会话”，确认侧栏已列出对应条约及版本 |
| 显示分析未完成或模型超时 | 展开“查看检索过程”定位失败节点；检索、证据检查和生成分别耗时，不要只增加检索轮数 |
| 端口被占用 | 停止占用进程或修改端口，保持 `API_BASE_URL` 一致 |

诊断、解析和索引报告自动写入 `reports/`，运行日志写入 `data/logs/`。就绪检查只读取本地状态，不请求模型。

## 命令行与测试

```bash
.venv/bin/python -m patent_agent.cli ask '专利法第二十二条规定了什么？'
.venv/bin/python -m patent_agent.cli ask '巴黎公约第四条之二如何规定不同国家专利的独立性？' --mode lookup
# 将 SESSION_ID 替换为上次返回的 session_id
.venv/bin/python -m patent_agent.cli ask '它对实用新型的要求呢？' --session SESSION_ID

.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
```

语料相关测试需要先准备上述 PDF 并构建索引。以下可选评测会调用真实模型，产生 API 用量：

```bash
.venv/bin/python scripts/evaluate.py
.venv/bin/python scripts/summarize_evaluation.py
# 单题回归（仍调用真实模型），独立保存结果
.venv/bin/python scripts/evaluate.py --case q24 --output reports/q24.jsonl --force
```

题集位于 `eval/cases.json`；结果写入 `reports/evaluation.jsonl`，按快照和题目 ID 断点续跑。完整重跑前应归档旧结果。汇总工具按快照分别统计引用完整性、检索预算和预设行为检查，不将这些指标作为法律语义准确率。跨文件案例可通过 `required_locators` 指定必须引用的依据；补查清单还检查各目标是否完整纳入上下文。

## 功能与目录

模型通过 LangGraph 选择精确定位或语义检索工具，检查证据，必要时补查，生成回答并校验引用。证据检查将明确缺口整理为结构化清单，由程序执行批量精确补查（每批最多 8 项），并记录各目标的查询结果与上下文保留情况；语义检索同时返回相关章节目录辅助定位，目录不作为回答证据；无法确定位置时通过语义检索补查。普通查询默认最多两轮检索；题目分析默认三轮，仍受每轮两个工具、八次模型调用及总时间预算约束。普通查询最多一次引用修复，题目分析最多一次结构修复，引用不通过则保留为受限回答。最终证据最多 16 个完整片段、10,000 字符，超出预算或未找到的依据会保留为回答限制。页面提供证据、完整条文、PDF 对应页及执行过程查看。

```text
Chinese_patent_agent/
├── .env.example         # 配置模板
├── config/              # 资料路径、哈希与版本清单
├── src/patent_agent/    # 解析、索引、检索、工作流、API 和页面
├── scripts/             # 下载、材料导入、启动与评测命令
├── docs/                # 题目分析设计和国际资料说明
├── tests/               # 自动化测试
├── eval/                # 评测题集
├── data/                # 本地资料、索引、会话及日志，运行时生成
├── reports/             # 运行报告，按需生成
├── pyproject.toml       # Python 依赖和工具配置
└── uv.lock              # 依赖锁定文件
```

资料版本以清单为准，系统不自动更新法规。PDF 文本解析不完整覆盖图片、化学结构、复杂公式和表格。引用校验检查来源、快照与逐字引文，回答解释是否充分受原文支持仍需结合证据判断。
