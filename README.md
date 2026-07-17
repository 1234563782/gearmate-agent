# GearMate Agent

GearMate Agent 是面向 RentFlow 电子设备租赁平台的智能选品与报价服务。它接收用户的自然语言请求，将请求解析为受控业务动作，并调用 RentFlow 完成商品搜索、商品详情查询、租期库存查询、设备组合推荐、正式报价和当前用户订单查询。

服务不会预占或释放库存，也不会创建订单。正式报价不会锁定库存，最终可租状态以 RentFlow 后续创建预占时为准。

## 已实现能力

- 使用 LangGraph 编排预处理、模型调用、工具执行、事实校验和结果输出。
- 解析商品搜索、商品详情、库存查询、报价、订单列表和多设备场景等意图。
- 支持直播、活动拍摄和商务会议三个可配置的设备组合场景。
- 持久化租期、待完成搜索、待完成租赁操作、最近商品结果和结构化场景需求。
- 使用“历史摘要 + 最近消息 + Token Budget”控制长对话上下文。
- 对回答中的商品 ID、金额、数量和商品名称进行工具事实一致性校验。
- 通过 PostgreSQL 持久化运行状态和事件，并使用 SSE、心跳和 `Last-Event-ID` 输出事件流。
- 使用 RS256 JWT 验证用户身份，并将用户访问令牌透传给需要身份的 RentFlow 接口。
- 可选启用 OpenAI Compatible Embedding 和 pgvector 商品语义检索。
- 定期清理超过保留期限的非活跃会话，并在启动时标记过期的活动运行。

## 系统结构

```mermaid
flowchart LR
    Client["客户端"] --> API["FastAPI / JWT"]
    API --> Coordinator["RunCoordinator"]
    Coordinator --> Resolver["意图、租期与需求解析"]
    Resolver --> Graph["LangGraph Agent"]
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
  -> 租期/场景需求补全
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
- 最高日租金
- 可选租期

对于“每天大约 200 元”一类偏好价格，服务会按商品日租金与目标价格的距离重新排序；对于“每天不超过 200 元”一类硬限制，则执行最高日租金过滤。

### 语义检索

语义检索默认关闭。启用后，服务会：

1. 从 RentFlow 拉取商品详情和动态用途目录。
2. 将设备角色、品牌、型号、名称、描述和用途组成检索文本。
3. 通过 OpenAI Compatible Embedding 接口生成 `1024` 维向量。
4. 在 pgvector 中使用 HNSW 余弦向量索引查询候选。
5. 融合向量相似度与名称、品牌、型号的词法匹配分数，对候选重新排序。
6. 从 RentFlow 回填候选商品的最新详情；提供租期时同时查询实时库存。

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
| `GEARMATE_RENTAL_PERIOD_MAX_ADVANCE_DAYS` | `90` | 允许的最远租赁开始日期 |
| `GEARMATE_CONVERSATION_RETENTION_HOURS` | `24` | 非活跃会话保留时间 |
| `GEARMATE_SSE_HEARTBEAT_SECONDS` | `15` | SSE 心跳间隔 |
| `GEARMATE_CATALOG_SYNC_INTERVAL_SECONDS` | `900` | 语义目录正常同步间隔 |

完整配置及默认值见 `.env.example` 和 `src/gearmate/config.py`。

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
  -d '{"message":"想租一台适合剪辑视频的电脑，每天大约 200 元"}'
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
```

## 测试与质量检查

```bash
uv run pytest --basetemp=.pytest-tmp/readme
uv run ruff check src tests
uv run mypy src
```

测试覆盖动作解析、租期策略、会话记忆、场景计划、商品检索、语义检索降级、推荐展示、订单工具、API 和会话清理等路径。

## 项目结构

```text
src/gearmate/
├── agent/              # LangGraph 工作流和运行协调
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
├── recommendations.py  # 商品推荐展示结构
├── rental_period.py    # 租期解析和策略校验
└── requirements.py     # 多设备场景需求解析

alembic/                # 数据库迁移
tests/                  # 自动化测试
compose.yaml            # 本地 PostgreSQL 基础配置
```

## 业务边界

- 库存查询必须提供精确商品和完整租期。
- 正式报价必须提供精确商品和完整租期，但报价不会锁定库存。
- Agent 只能查询当前 JWT 所属用户的订单。
- 商品、库存、价格、押金和报价金额只能来自 RentFlow 工具结果。
- 工具失败时不会使用模型常识补全业务事实。
