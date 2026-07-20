from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

from gearmate.actions import AgentAction, AgentActionResolver
from gearmate.catalog import CatalogSearchRepository
from gearmate.config import Settings
from gearmate.llm.factory import build_chat_model
from gearmate.persistence.database import Database

INTENTS = (
    "chat",
    "product_search",
    "product_detail",
    "availability",
    "quote",
    "order_list",
    "scenario_continue",
)
NORMALIZATION_FIELDS = (
    "equipmentRole",
    "brand",
    "model",
    "useCaseId",
    "orderStatus",
)


@dataclass(frozen=True, slots=True)
class EvalCase:
    case_id: str
    message: str
    expected_action: str
    expected_tool: str | None
    normalization: dict[str, str | None] | None


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        normalization = payload.get("normalization")
        if normalization is not None and set(normalization) != set(NORMALIZATION_FIELDS):
            raise ValueError(
                f"{path}:{line_number}: normalization must label all supported fields"
            )
        cases.append(
            EvalCase(
                case_id=str(payload["caseId"]),
                message=str(payload["message"]),
                expected_action=str(payload["expectedAction"]),
                expected_tool=(
                    str(payload["expectedTool"])
                    if payload.get("expectedTool") is not None
                    else None
                ),
                normalization=normalization,
            )
        )
    duplicate_ids = [
        case_id
        for case_id, count in Counter(c.case_id for c in cases).items()
        if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"Duplicate case IDs: {duplicate_ids}")
    unknown_intents = sorted({case.expected_action for case in cases} - set(INTENTS))
    if unknown_intents:
        raise ValueError(f"Unknown expected intents: {unknown_intents}")
    return cases


def planned_tool(action: AgentAction | None) -> str | None:
    if action is None:
        return None
    if action.action == "product_search":
        return "search_products"
    if action.action == "product_detail":
        return "get_product" if action.product_id is not None else None
    if action.action == "availability":
        return "check_availability" if action.product_id is not None else None
    if action.action == "quote":
        return "create_quote" if action.product_id is not None else None
    if action.action == "order_list":
        return "list_orders"
    if action.action == "scenario_continue":
        return "set_rental_requirements"
    return None


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def intent_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    per_intent: dict[str, dict[str, int | float | None]] = {}
    f1_values: list[float] = []
    for intent in INTENTS:
        true_positive = sum(
            result["expectedAction"] == intent and result["predictedAction"] == intent
            for result in results
        )
        false_positive = sum(
            result["expectedAction"] != intent and result["predictedAction"] == intent
            for result in results
        )
        false_negative = sum(
            result["expectedAction"] == intent and result["predictedAction"] != intent
            for result in results
        )
        precision = safe_ratio(true_positive, true_positive + false_positive)
        recall = safe_ratio(true_positive, true_positive + false_negative)
        f1 = (
            round(2 * precision * recall / (precision + recall), 6)
            if precision is not None and recall is not None and precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_intent[intent] = {
            "support": sum(result["expectedAction"] == intent for result in results),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    correct = sum(result["intentCorrect"] for result in results)
    return {
        "accuracy": safe_ratio(correct, len(results)),
        "macroF1": round(sum(f1_values) / len(f1_values), 6),
        "correct": correct,
        "total": len(results),
        "perIntent": per_intent,
    }


def normalization_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [result for result in results if result["expectedNormalization"] is not None]
    case_exact = 0
    positive_correct = 0
    positive_total = 0
    false_positive = 0
    expected_null_total = 0
    per_field: dict[str, dict[str, int | float | None]] = {}

    for field in NORMALIZATION_FIELDS:
        field_correct = 0
        for result in labeled:
            expected = result["expectedNormalization"][field]
            predicted = result["predictedNormalization"][field]
            field_correct += expected == predicted
            if expected is not None:
                positive_total += 1
                positive_correct += expected == predicted
            else:
                expected_null_total += 1
                false_positive += predicted is not None
        per_field[field] = {
            "accuracy": safe_ratio(field_correct, len(labeled)),
            "correct": field_correct,
            "total": len(labeled),
        }

    for result in labeled:
        case_exact += result["expectedNormalization"] == result["predictedNormalization"]

    return {
        "caseExactMatch": safe_ratio(case_exact, len(labeled)),
        "caseExactCorrect": case_exact,
        "labeledCases": len(labeled),
        "positiveFieldAccuracy": safe_ratio(positive_correct, positive_total),
        "positiveFieldCorrect": positive_correct,
        "positiveFieldTotal": positive_total,
        "falsePositiveRate": safe_ratio(false_positive, expected_null_total),
        "falsePositiveCount": false_positive,
        "expectedNullTotal": expected_null_total,
        "perField": per_field,
    }


def tool_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(result["toolCorrect"] for result in results)
    by_tool: dict[str, dict[str, int | float | None]] = {}
    expected_tools = sorted({str(result["expectedTool"]) for result in results})
    for tool in expected_tools:
        matching = [result for result in results if str(result["expectedTool"]) == tool]
        tool_correct = sum(result["toolCorrect"] for result in matching)
        by_tool[tool] = {
            "accuracy": safe_ratio(tool_correct, len(matching)),
            "correct": tool_correct,
            "total": len(matching),
        }
    return {
        "plannedToolAccuracy": safe_ratio(correct, len(results)),
        "correct": correct,
        "total": len(results),
        "byExpectedTool": by_tool,
        "definition": (
            "根据 AgentAction 推导的首个计划工具路由；"
            "该指标不衡量 RentFlow 工具接口执行成功率。"
        ),
    }


def _markdown_text(value: object) -> str:
    if value is None:
        return "空"
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _normalization_difference(result: dict[str, Any]) -> str:
    expected = result["expectedNormalization"]
    predicted = result["predictedNormalization"]
    if expected is None or predicted is None:
        return ""
    differences = [
        f"{field}: {_markdown_text(expected[field])} -> {_markdown_text(predicted[field])}"
        for field in NORMALIZATION_FIELDS
        if expected[field] != predicted[field]
    ]
    return "<br>".join(differences)


def render_failure_markdown(report: dict[str, Any]) -> str:
    failures = report["failures"]
    intent_failures = sum(not result["intentCorrect"] for result in failures)
    normalization_failures = sum(
        result["normalizationCorrect"] is False for result in failures
    )
    tool_failures = sum(not result["toolCorrect"] for result in failures)
    request_failures = sum(result["error"] is not None for result in failures)
    lines = [
        "# 动作路由失败案例",
        "",
        f"- 测评时间：`{report['metadata']['generatedAt']}`",
        f"- 模型：`{report['metadata']['modelId']}`",
        f"- 总样本：`{report['metadata']['caseCount']}`",
        f"- 失败记录：`{len(failures)}`",
        f"- 意图失败：`{intent_failures}`",
        f"- 归一化失败：`{normalization_failures}`",
        f"- 工具路由失败：`{tool_failures}`",
        f"- 模型请求失败：`{request_failures}`",
        "",
        "> 一条案例可能同时存在意图失败和工具路由失败，分类数量不能直接相加。",
        "",
        "| caseId | 用户输入 | 失败类型 | 期望结果 | 实际结果 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in failures:
        failure_types: list[str] = []
        expected_parts: list[str] = []
        predicted_parts: list[str] = []
        if not result["intentCorrect"]:
            failure_types.append("意图识别")
            expected_parts.append(f"意图: {result['expectedAction']}")
            predicted_parts.append(f"意图: {result['predictedAction']}")
        if result["normalizationCorrect"] is False:
            failure_types.append("实体归一化")
            expected_parts.append("归一化字段见实际差异")
            predicted_parts.append(_normalization_difference(result))
        if not result["toolCorrect"]:
            failure_types.append("工具路由")
            expected_parts.append(f"工具: {_markdown_text(result['expectedTool'])}")
            predicted_parts.append(f"工具: {_markdown_text(result['predictedTool'])}")
        if result["error"] is not None:
            failure_types.append("模型请求")
            expected_parts.append("请求成功")
            predicted_parts.append(f"异常: {_markdown_text(result['error'])}")
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_text(result["caseId"]),
                    _markdown_text(result["message"]),
                    "、".join(failure_types),
                    "<br>".join(expected_parts),
                    "<br>".join(predicted_parts),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## 查看完整预测",
            "",
            "本文件只展示不一致字段。完整的动作载荷、Token、耗时和错误信息，"
            "请查看同名 JSON 报告中的 `failures` 或 `results`。",
            "",
        )
    )
    return "\n".join(lines)


async def evaluate_case(
    case: EvalCase,
    *,
    resolver: AgentActionResolver,
    model: Any,
    vocabulary: Any,
    max_output_tokens: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    started = monotonic()
    action: AgentAction | None = None
    error: str | None = None
    usage: dict[str, int] = {"inputTokens": 0, "outputTokens": 0}
    try:
        async with semaphore:
            resolution = await resolver.resolve(
                message=case.message,
                history=(),
                current_scenario_id=None,
                pending_product_search=None,
                pending_rental_action=None,
                model=model,
                max_output_tokens=max_output_tokens,
                catalog_vocabulary=vocabulary,
            )
        action = resolution.action
        usage = {
            "inputTokens": resolution.usage.input_tokens,
            "outputTokens": resolution.usage.output_tokens,
        }
    except Exception as caught:  # The report must retain failed cases instead of aborting.
        error = f"{type(caught).__name__}: {caught}"

    predicted_payload = action.model_dump(mode="json", by_alias=True) if action else {}
    predicted_action = action.action if action else "__error__"
    predicted_normalization = (
        {field: predicted_payload.get(field) for field in NORMALIZATION_FIELDS}
        if case.normalization is not None
        else None
    )
    predicted_tool = planned_tool(action)
    return {
        "caseId": case.case_id,
        "message": case.message,
        "expectedAction": case.expected_action,
        "predictedAction": predicted_action,
        "intentCorrect": predicted_action == case.expected_action,
        "expectedNormalization": case.normalization,
        "predictedNormalization": predicted_normalization,
        "normalizationCorrect": (
            predicted_normalization == case.normalization
            if case.normalization is not None
            else None
        ),
        "expectedTool": case.expected_tool,
        "predictedTool": predicted_tool,
        "toolCorrect": predicted_tool == case.expected_tool,
        "predictedActionPayload": predicted_payload or None,
        "usage": usage,
        "durationMs": round((monotonic() - started) * 1000),
        "error": error,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = Settings()
    cases = load_cases(args.dataset)
    if args.limit is not None:
        cases = cases[: args.limit]
    database = Database(settings)
    model = build_chat_model(settings)
    try:
        vocabulary = await CatalogSearchRepository(database.session_factory).vocabulary()
        resolver = AgentActionResolver(settings.equipment_roles)
        semaphore = asyncio.Semaphore(args.concurrency)
        results = await asyncio.gather(
            *(
                evaluate_case(
                    case,
                    resolver=resolver,
                    model=model,
                    vocabulary=vocabulary,
                    max_output_tokens=settings.action_resolution_max_output_tokens,
                    semaphore=semaphore,
                )
                for case in cases
            )
        )
    finally:
        await model.close()
        await database.dispose()

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for result in results:
        confusion[result["expectedAction"]][result["predictedAction"]] += 1
    total_input_tokens = sum(result["usage"]["inputTokens"] for result in results)
    total_output_tokens = sum(result["usage"]["outputTokens"] for result in results)
    return {
        "metadata": {
            "generatedAt": datetime.now(UTC).isoformat(),
            "dataset": str(args.dataset),
            "modelId": settings.model_id,
            "caseCount": len(results),
            "concurrency": args.concurrency,
            "catalogVocabulary": {
                "equipmentRoles": len(vocabulary.equipment_roles),
                "brands": len(vocabulary.brands),
                "models": len(vocabulary.models),
                "aliases": len(vocabulary.aliases),
            },
            "usage": {
                "inputTokens": total_input_tokens,
                "outputTokens": total_output_tokens,
            },
            "failedRequests": sum(result["error"] is not None for result in results),
        },
        "metrics": {
            "intent": intent_metrics(results),
            "normalization": normalization_metrics(results),
            "toolSelection": tool_metrics(results),
        },
        "confusionMatrix": {key: dict(value) for key, value in confusion.items()},
        "failures": [
            result
            for result in results
            if not result["intentCorrect"]
            or result["normalizationCorrect"] is False
            or not result["toolCorrect"]
            or result["error"] is not None
        ],
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测评动作意图、目录实体归一化和首工具路由。"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/action_routing_cases.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    report = asyncio.run(run(args))
    output = args.output or Path("evals/results/action_routing_latest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    failure_output = output.with_name(f"{output.stem}_failures.md")
    failure_output.write_text(render_failure_markdown(report), encoding="utf-8")
    metrics = report["metrics"]
    print(f"完整报告：{output}")
    print(f"失败案例：{failure_output}")
    print(
        "意图识别："
        f"准确率={metrics['intent']['accuracy']}，"
        f"Macro-F1={metrics['intent']['macroF1']}"
    )
    print(
        "实体归一化："
        f"整例准确率={metrics['normalization']['caseExactMatch']}，"
        f"非空字段准确率={metrics['normalization']['positiveFieldAccuracy']}，"
        f"错误归一率={metrics['normalization']['falsePositiveRate']}"
    )
    print(
        "工具选择："
        f"首工具路由准确率={metrics['toolSelection']['plannedToolAccuracy']}"
    )
    print(f"模型请求失败：{report['metadata']['failedRequests']}")


if __name__ == "__main__":
    main()
