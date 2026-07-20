# 动作路由评测

本目录用于离线测评 GearMate 的动作解析能力。评测使用人工标注的 JSONL 数据集，对实际配置的动作模型发起请求，然后比较模型输出与标准答案。

## 测评内容

当前测评包含三部分：

1. **意图识别**：计算准确率、Macro-F1，以及每种意图的 Precision、Recall 和 F1。
2. **目录实体归一化**：计算整例完全匹配率、非空字段准确率、错误归一率和各字段准确率。
3. **首工具路由**：根据解析后的 `AgentAction` 推导下一步应该进入的工具，计算路由准确率。

首工具路由不是 RentFlow HTTP 执行成功率。它只判断动作解析结果能否路由到正确的第一个工具，不会实际调用 RentFlow，也不统计工具接口是否成功返回。

## 数据集

当前 [action_routing_cases.jsonl](action_routing_cases.jsonl) 包含 200 条单轮样本：

| 意图 | 样本数 |
| --- | ---: |
| `chat` | 24 |
| `product_search` | 64 |
| `product_detail` | 20 |
| `availability` | 24 |
| `quote` | 24 |
| `order_list` | 20 |
| `scenario_continue` | 24 |

其中 84 条商品搜索和订单查询样本包含字段级归一化标签。归一化字段包括：

- `equipmentRole`
- `brand`
- `model`
- `useCaseId`
- `orderStatus`

## 运行方式

在项目根目录执行：

```bash
python evals/evaluate_action_routing.py
```

可以限制并发数、只运行前几条或指定输出位置：

```bash
python evals/evaluate_action_routing.py \
  --concurrency 4 \
  --limit 20 \
  --output evals/results/action_routing_latest.json
```

运行需要：

- `.env` 中已经配置聊天模型；
- GearMate PostgreSQL 可访问；
- PostgreSQL 中已有目录角色、品牌、型号和别名数据。

## 输出文件

默认生成两个文件：

- `evals/results/action_routing_latest.json`：完整机器可读报告，包含指标、混淆矩阵、失败列表和全部 200 条预测。
- `evals/results/action_routing_latest_failures.md`：中文失败案例报告，逐条列出输入、失败类型、期望结果和实际结果。

在 JSON 报告中：

- `metrics` 是汇总指标；
- `confusionMatrix` 是意图混淆矩阵；
- `failures` 只保存至少一项失败的案例；
- `results` 保存所有案例的完整预测。

## 如何定位失败案例

失败案例有三种查看方式：

1. 直接打开 `action_routing_latest_failures.md` 阅读中文列表。
2. 在 `action_routing_latest.json` 的 `failures` 数组中查看完整动作载荷、Token 和耗时。
3. 使用失败记录中的 `caseId` 搜索 `action_routing_cases.jsonl`，找到该案例的原始人工标签。

例如：

```bash
rg 'scenario-001' evals/action_routing_cases.jsonl
```

## 指标解释

### 意图准确率

```text
意图准确率 = 意图预测正确的样本数 / 总样本数
```

### Macro-F1

先分别计算 7 种意图的 F1，再取算术平均。它不会因为商品搜索样本较多而掩盖场景意图表现较差的问题。

### 归一化整例准确率

一条样本的所有已标注归一化字段都与标准答案一致，才算整例正确。

### 非空字段准确率

只统计标准答案非空的字段，用来判断应该归一化的实体是否映射到了正确标准值。

### 错误归一率

标准答案应为空，但模型输出了品牌、型号、用途等值时，记为错误归一化。

### 首工具路由准确率

```text
首工具路由准确率 = 路由到正确首工具的样本数 / 总样本数
```

当前工具路由由动作结果确定，因此意图误判通常会同时造成工具路由错误。

## 当前边界

- 当前 200 条数据是单轮离线样本，不测多轮上下文继承。
- 当前工具指标不测 RentFlow 网络请求、参数 Schema 或业务返回结果。
- 数据集以现有商品目录和别名为基础，不能直接代表线上真实用户分布。
- 每次更换模型、Prompt、目录词表或动作后处理逻辑后，都应重新运行并保存报告。
