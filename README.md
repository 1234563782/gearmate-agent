# GearMate Agent

GearMate Agent 是面向 RentFlow 数码潮玩商城的智能选购服务。它接收用户的自然语言请求，将请求解析为受控商城动作，并调用 RentFlow 完成商品搜索、商品与 SKU 详情查询、可售库存查询、购买选项准备和当前用户商城订单查询。

服务不会替用户创建订单、支付、取消订单或确认收货。购买动作只把可信的商品、SKU 和数量交给前端，由用户在结算页确认收货信息后提交订单。

## 已实现能力

- 使用显式异步状态机编排预处理、模型调用、工具执行、事实校验和结果输出。
- 解析商品搜索、商品详情、SKU 库存、购买准备、商城订单列表和订单详情等意图。
- 商品搜索支持品类、用途、品牌、型号、目标购买价和最高购买价。
- 持久化待完成搜索、最近商品结果和会话摘要，用于“第一个”“更看重性能”等跨轮表达。
- 使用“历史摘要 + 最近消息 + Token Budget”控制长对话上下文。
- 可选启用以 `user_id` 隔离的长期偏好记忆，支持跨会话读取、纠正和删除。
- 阻止回答泄露内部 ID，并对金额、数量和商品名称进行工具事实一致性校验。
- 通过 PostgreSQL 持久化运行状态和事件，并使用 SSE、心跳和 `Last-Event-ID` 输出事件流。
- 使用 RS256 JWT 验证用户身份，并将用户访问令牌透传给需要身份的 RentFlow 接口。
- 可选启用 OpenAI Compatible Embedding 和 pgvector 商品语义检索。
- 定期清理超过保留期限的非活跃会话，并在启动时标记过期的活动运行。

## 系统结构

```mermaid
flowchart LR
    Client["客户端"] --> API["FastAPI / JWT"]
    API --> Coordinator["RunCoordinator"]
    Coordinator --> Resolver["商城动作与参数解析"]
    Resolver --> Graph["异步 Agent 状态机"]
    Graph --> Tools["受控工具注册表"]
    Tools --> RentFlow["RentFlow API"]
    Coordinator --> DB["PostgreSQL"]
    Graph --> DB
    DB --> SSE["SSE 事件流"]
    SSE --> Client
    Tools -. "可选" .-> Search["Embedding + pgvector"]
```

一次 Agent 运行的主流程为：

```text
用户消息
  -> 动作分类
  -> 会话状态合并与参数校验
  -> preprocess
  -> model <-> tools
  -> validate
  -> finalize
  -> 持久化运行结果和事件
```

## 检索实现

### 结构化检索

结构化检索直接调用 RentFlow 商品接口，支持以下条件：

- 关键词
- 设备角色
- 品牌与型号
- 动态用途 ID
- 品类 ID
- 最高购买价
- 目标购买价

对于“5000 元左右”一类偏好价格，服务按 SKU 售价与目标购买价的距离排序；对于“预算不超过 5000 元”一类硬限制，则执行最高购买价过滤。库存和售价始终以 RentFlow 本轮返回的 SKU 数据为准。

### 语义检索


语义检索默认关闭。启用后，服务会：

1. 从 RentFlow 拉取商品详情和动态用途目录。
2. 将设备角色、品牌、型号、名称、描述和用途组成检索文本。
3. 通过 OpenAI Compatible Embedding 接口生成 `1024` 维向量。
4. 在 pgvector 中使用 HNSW 余弦向量索引查询候选。
5. 融合向量相似度与名称、品牌、型号的词法匹配分数，对候选重新排序。
6. 从 RentFlow 回填候选商品、SKU 售价和可售库存。

语义检索支持设备角色、品牌、型号和动态用途过滤。语义服务异常或没有达到阈值的候选时，自动回退到 RentFlow 结构化检索。

当前词法部分是候选集上的规则评分，不是独立的全文检索或 BM25 召回。仓库也没有提供检索性能基准，因此不对 HNSW 的性能收益作量化声明。

## 环境要求

- Python `3.12`
- PostgreSQL，且数据库已安装 pgvector 扩展
- 可访问的 RentFlow 服务
- 支持 Chat Completions 和函数工具调用的 OpenAI Compatible 模型服务
- RentFlow 签发 JWT 所对应的 RS256 公钥
- 可选：支持 `1024` 维输出的 OpenAI Compatible Embedding 服务

> [!IMPORTANT]
> Alembic 迁移会执行 `CREATE EXTENSION IF NOT EXISTS vector`。仓库当前的 `compose.yaml` 使用标准 `postgres:17.5-alpine` 镜像，该镜像通常不包含 pgvector。运行迁移前，请使用已安装 pgvector 的 PostgreSQL 实例或自行在镜像中安装扩展。

## 本地启动

### 1. 安装依赖

项目包含 `uv.lock`，推荐使用 uv：

```bash
uv sync --extra dev
```

也可以使用已有的 Python 3.12 环境：

```bash
python -m pip install -e ".[dev]"
```

### 2. 配置环境变量

复制示例配置：

```powershell
Copy-Item .env.example .env
```

至少需要设置：

```dotenv
GEARMATE_DATABASE_URL=postgresql+asyncpg://gearmate:password@localhost:5432/gearmate
GEARMATE_RENTFLOW_BASE_URL=http://localhost:8080

GEARMATE_JWT_PUBLIC_KEY_PATH=C:/path/to/rentflow-public.pem
GEARMATE_JWT_ISSUER=rentflow-server
GEARMATE_JWT_AUDIENCE=rentflow-platform

GEARMATE_MODEL_BASE_URL=https://model-provider.example/v1
GEARMATE_MODEL_ID=your-chat-model
GEARMATE_MODEL_API_KEY=your-api-key
```

模型配置缺失时服务可以启动，但创建 Agent 运行会返回模型配置错误。JWT 公钥未配置时，受保护的业务接口返回 `503`。

### 3. 执行数据库迁移

```bash
uv run alembic upgrade head
```

未使用 uv 时执行：

```bash
alembic upgrade head
```

### 4. 启动服务

```bash
uv run gearmate
```

服务默认监听 `http://localhost:8000`：

- 健康检查：`GET http://localhost:8000/health`
- OpenAPI 文档：`http://localhost:8000/docs`
- OpenAPI JSON：`http://localhost:8000/openapi.json`

也可以直接通过 Uvicorn 启动：

```bash
uv run uvicorn gearmate.main:app --host 0.0.0.0 --port 8000
```

## 启用语义检索

在 `.env` 中配置：

```dotenv
GEARMATE_SEMANTIC_SEARCH_ENABLED=true
GEARMATE_EMBEDDING_BASE_URL=https://embedding-provider.example/v1
GEARMATE_EMBEDDING_MODEL_ID=your-embedding-model
GEARMATE_EMBEDDING_API_KEY=your-api-key
GEARMATE_EMBEDDING_DIMENSIONS=1024
GEARMATE_EMBEDDING_BATCH_SIZE=10
GEARMATE_CATALOG_SYNC_ON_STARTUP=true
```

如果 Embedding 服务和聊天模型共用地址或 API Key，可以省略 `GEARMATE_EMBEDDING_BASE_URL` 或 `GEARMATE_EMBEDDING_API_KEY`，服务会使用对应的模型配置。`GEARMATE_EMBEDDING_MODEL_ID` 必须单独配置。

目录同步只为内容发生变化的商品重新生成向量，并将 RentFlow 已删除的商品标记为非活动状态。启动同步失败后，后台任务会按重试间隔继续尝试。

## 主要配置

| 配置 | 默认值 | 说明 |
| --- | ---: | --- |
| `GEARMATE_RUN_TIMEOUT_SECONDS` | `180` | 单次 Agent 运行超时 |
| `GEARMATE_MAX_MODEL_ROUNDS` | `6` | 单次运行最大模型轮次 |
| `GEARMATE_MAX_TOOL_CALLS` | `10` | 单次运行最大工具调用数 |
| `GEARMATE_MAX_TOOL_CONCURRENCY` | `4` | 并发安全工具的并发上限 |
| `GEARMATE_CONTEXT_HISTORY_TOKEN_BUDGET` | `12000` | 历史上下文 Token 预算 |
| `GEARMATE_CONTEXT_SUMMARY_TRIGGER_TOKENS` | `8000` | 触发会话摘要的估算 Token 数 |
| `GEARMATE_USER_MEMORY_ENABLED` | `false` | 是否启用用户长期记忆读取与抽取 |
| `GEARMATE_USER_MEMORY_MODE` | `off` | 长期记忆模式：`off`、`shadow` 或 `active` |
| `GEARMATE_USER_MEMORY_RETRIEVAL_LIMIT` | `10` | 单轮最多读取的有效长期记忆数 |
| `GEARMATE_USER_MEMORY_MIN_CONFIDENCE` | `0.85` | 模型抽取候选的最低置信度 |
| `GEARMATE_USER_MEMORY_RETENTION_DAYS` | `180` | 用户长期记忆的默认有效期 |
| `GEARMATE_INTENT_PRE_ROUTER_MODE` | `off` | 确定性意图预路由模式：`off`、`shadow` 或 `enforce` |
| `GEARMATE_INTENT_PRE_ROUTER_PURE_SOCIAL_ENABLED` | `true` | 启用纯寒暄候选规则 |
| `GEARMATE_CONVERSATION_RETENTION_HOURS` | `24` | 非活跃会话保留时间 |
| `GEARMATE_SSE_HEARTBEAT_SECONDS` | `15` | SSE 心跳间隔 |
| `GEARMATE_CATALOG_SYNC_INTERVAL_SECONDS` | `900` | 语义目录正常同步间隔 |
| `GEARMATE_CHAT_MODEL_MAX_CONCURRENCY` | `6` | 单进程聊天模型总并发上限 |
| `GEARMATE_ACTION_MODEL_MAX_CONCURRENCY` | `4` | 商城动作和参数解析并发上限 |
| `GEARMATE_MAIN_MODEL_MAX_CONCURRENCY` | `3` | 主回答生成并发上限 |
| `GEARMATE_BACKGROUND_MODEL_MAX_CONCURRENCY` | `1` | 会话摘要并发上限 |
| `GEARMATE_CHAT_QUEUE_CAPACITY` | `40` | 聊天模型有界等待队列容量 |
| `GEARMATE_EMBEDDING_MAX_CONCURRENCY` | `2` | 单进程 Embedding 总并发上限 |
| `GEARMATE_EMBEDDING_ONLINE_MAX_CONCURRENCY` | `2` | 在线语义查询并发上限 |
| `GEARMATE_EMBEDDING_REFRESH_MAX_CONCURRENCY` | `1` | 目录刷新 Embedding 并发上限 |

完整配置及默认值见 `.env.example` 和 `src/gearmate/config.py`。

聊天模型和 Embedding 模型使用相互独立的有界队列、并发信号量、RPM/TPM
令牌桶、指数退避和断路器。动作解析、主回答、后台摘要以及在线查询、目录刷新分别使用
独立 workload lane，避免目录刷新或摘要任务占满在线请求容量。OpenAI SDK 自带重试已关闭，
由统一治理层处理 `429`、连接/超时和 `5xx`，并遵循服务端 `Retry-After`。

### 用户长期记忆

长期记忆与 `ConversationMemoryService` 管理的会话状态相互独立。待完成搜索、最近搜索和
摘要继续按 `conversation_id` 保存并随会话保留策略清理；长期记忆按 JWT 中的
`user_id` 保存，不会随来源会话级联删除。

当前仅允许记录用户明确表达的稳定偏好：偏好或排斥品牌、常用设备角色、常用场景和语言。
临时预算、商品选择、售价、库存、订单状态、物流、内部 ID 和敏感个人信息
不会写入长期记忆。历史偏好只用于提示和商品软重排，当前消息始终优先；所有动态业务事实
仍通过 RentFlow 工具实时查询。

品牌、设备角色和使用场景会先按 `catalog_aliases` 生成稳定事实身份，同一用户的记忆写入
在事务内串行处理，避免同义表达、Unicode 大小写差异或并发写入留下互相冲突的有效偏好。
过期记录会进入 `EXPIRED` 状态，不再参与撤销、替换或冲突覆盖。`valid_from` 表示该事实
首次生效时间，`last_confirmed_at` 和 `source_*` 表示最近一次确认及其来源。

建议先使用 `shadow` 模式评估抽取结果。该模式会执行候选抽取但不写入数据库，也不会影响
Agent 回答；评测确认后再切换为 `active`。

这些限制在单个 GearMate 进程内生效。当前单实例部署不需要 Redis；扩展为多个 GearMate
实例时，应将全局 RPM/TPM 配额和分布式并发租约迁移到 Redis，各实例仍保留本地有界队列与断路器。

## API 使用流程

除健康检查外，业务接口都要求：

```http
Authorization: Bearer <RentFlow JWT>
```

典型调用顺序如下。

### 1. 创建会话

```bash
curl -X POST http://localhost:8000/api/v1/conversations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title":"设备咨询","timezone":"Asia/Shanghai"}'
```

### 2. 创建 Agent 运行

```bash
curl -X POST http://localhost:8000/api/v1/conversations/<conversation-id>/runs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"想买一台适合剪辑视频的电脑，预算 8000 元左右"}'
```

该接口返回 `202 Accepted` 和 `runId`。同一个会话同时只允许一个活动运行，冲突时返回 `409`。

### 3. 订阅运行事件

```bash
curl -N http://localhost:8000/api/v1/runs/<run-id>/events?after=0 \
  -H "Authorization: Bearer $TOKEN"
```

重新连接时可以传递最后收到的事件序号：

```http
Last-Event-ID: 12
```

### 4. 查询运行或消息

```text
GET /api/v1/runs/{run_id}
GET /api/v1/conversations/{conversation_id}/messages?limit=100
POST /api/v1/runs/{run_id}/cancel
GET /api/v1/me/memories
PATCH /api/v1/me/memories/{memory_id}
DELETE /api/v1/me/memories/{memory_id}
DELETE /api/v1/me/memories
```

## 测试与质量检查

```bash
uv run pytest --basetemp=.pytest-tmp/readme
uv run ruff check src tests
uv run mypy src
```

测试覆盖动作解析、会话记忆、用户长期记忆、商品检索、语义检索降级、SKU 展示、商城订单工具、API 和会话清理等路径。仓库中尚有迁移期遗留模块的隔离测试，但它们不属于当前商城入口。

## 项目结构

```text
src/gearmate/
├── agent/              # 异步状态机和运行协调
├── api/                # FastAPI 路由
├── auth/               # JWT 身份验证
├── llm/                # OpenAI Compatible 模型适配
├── persistence/        # SQLAlchemy 模型和仓储
├── prompts/            # 系统提示词与场景配置
├── rentflow/           # RentFlow HTTP 客户端
├── streaming/          # SSE 编码
├── tools/              # 工具定义、参数模型和执行注册表
├── validation/         # 工具事实校验
├── actions.py          # 当前轮动作解析
├── catalog.py          # 商品目录同步与语义检索
├── memory.py           # 会话上下文和摘要
├── user_memory.py      # 用户长期偏好抽取、读取和治理
└── recommendations.py  # 商品推荐展示结构

alembic/                # 数据库迁移
tests/                  # 自动化测试
compose.yaml            # 本地 PostgreSQL 基础配置
```

## 业务边界

- 商品发现以商品为单位，售价和库存以 SKU 为单位。
- 购买准备只返回商品、SKU 和数量，不会创建或支付订单。
- Agent 只能查询当前 JWT 所属用户的订单。
- 商品、SKU、库存、售价、订单状态和物流只能来自 RentFlow 工具结果。
- 工具失败时不会使用模型常识补全业务事实。
