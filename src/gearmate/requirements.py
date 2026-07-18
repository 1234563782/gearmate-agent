import json
import tomllib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.alias_generators import to_camel

from gearmate.llm.port import ChatModelPort
from gearmate.llm.types import (
    ModelMessage,
    ModelRequest,
    ModelToolDefinition,
    ModelUsage,
)

REQUIREMENTS_TOOL_NAME = "set_rental_requirements"
RequirementAnswer = str | int | bool
FieldKind = Literal["boolean", "choice", "integer", "text"]
EquipmentRole = str


class RequirementModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )


class RentalRequirements(RequirementModel):
    scenario_id: str = Field(min_length=1, max_length=64)
    daily_budget: Decimal | None = Field(default=None, gt=0, max_digits=10)
    answers: dict[str, RequirementAnswer] = Field(default_factory=dict)


class EquipmentNeed(RequirementModel):
    role: str = Field(min_length=1, max_length=64)
    keyword: str = Field(min_length=1, max_length=128)
    quantity: int = Field(ge=1, le=100)


@dataclass(frozen=True, slots=True)
class ScenarioField:
    name: str
    question: str
    kind: FieldKind
    required: bool
    options: tuple[str, ...]
    signals: tuple[str, ...]
    minimum: int | None
    maximum: int | None
    required_when_field: str | None
    required_when_equals: RequirementAnswer | None

    def is_required(self, answers: dict[str, RequirementAnswer]) -> bool:
        if self.required_when_field is None:
            return self.required
        return answers.get(self.required_when_field) == self.required_when_equals

    def valid(self, value: RequirementAnswer | None) -> bool:
        if value is None or (isinstance(value, str) and not value.strip()):
            return False
        if self.kind == "boolean":
            return isinstance(value, bool)
        if self.kind == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return False
            if self.minimum is not None and value < self.minimum:
                return False
            return self.maximum is None or value <= self.maximum
        if self.kind == "choice":
            return isinstance(value, str) and value in self.options
        return isinstance(value, str)


@dataclass(frozen=True, slots=True)
class ScenarioEquipment:
    role: str
    keyword: str
    quantity: int
    quantity_field: str | None
    include_when_field: str | None
    include_when_equals: RequirementAnswer | None

    def included(self, answers: dict[str, RequirementAnswer]) -> bool:
        if self.include_when_field is None:
            return True
        actual = answers.get(self.include_when_field)
        if self.include_when_equals is None:
            return actual is True
        return actual == self.include_when_equals

    def resolved_quantity(self, answers: dict[str, RequirementAnswer]) -> int:
        if self.quantity_field is None:
            return self.quantity
        value = answers.get(self.quantity_field)
        return value if isinstance(value, int) and not isinstance(value, bool) else self.quantity


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    scenario_id: str
    name: str
    aliases: tuple[str, ...]
    fields: tuple[ScenarioField, ...]
    equipment: tuple[ScenarioEquipment, ...]

    def extraction_schema(self) -> dict[str, object]:
        return {
            "scenarioId": self.scenario_id,
            "name": self.name,
            "fields": [
                {
                    "name": field.name,
                    "kind": field.kind,
                    "options": list(field.options),
                    "question": field.question,
                }
                for field in self.fields
            ],
        }


@dataclass(frozen=True, slots=True)
class ScenarioPlan:
    requirements: RentalRequirements
    scenario_name: str
    equipment_needs: tuple[EquipmentNeed, ...]
    missing_fields: tuple[str, ...]
    clarification: str | None

    @property
    def ready(self) -> bool:
        return not self.missing_fields

    def model_context(self) -> dict[str, object]:
        return {
            "scenarioId": self.requirements.scenario_id,
            "scenarioName": self.scenario_name,
            "dailyBudget": (
                str(self.requirements.daily_budget)
                if self.requirements.daily_budget is not None
                else None
            ),
            "answers": self.requirements.answers,
            "equipmentNeeds": [
                item.model_dump(mode="json", by_alias=True) for item in self.equipment_needs
            ],
        }


@dataclass(frozen=True, slots=True)
class RequirementsResolution:
    plan: ScenarioPlan | None
    clarification: str | None
    usage: ModelUsage


class ScenarioCatalog:
    def __init__(
        self,
        definitions: tuple[ScenarioDefinition, ...],
        common_followup_signals: tuple[str, ...] = (),
    ) -> None:
        if not definitions:
            raise ValueError("Scenario catalog must not be empty")
        scenario_ids = [definition.scenario_id for definition in definitions]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("Scenario ids must be unique")
        if any(not definition.aliases for definition in definitions):
            raise ValueError("Every scenario must define at least one alias")
        self._definitions = {definition.scenario_id: definition for definition in definitions}
        self._common_followup_signals = common_followup_signals

    @classmethod
    def load_default(cls) -> "ScenarioCatalog":
        path = Path(__file__).parent / "prompts" / "scenarios.toml"
        return cls.from_toml(path)

    @classmethod
    def from_toml(cls, path: Path) -> "ScenarioCatalog":
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        definitions: list[ScenarioDefinition] = []
        for raw in payload.get("scenarios", []):
            fields: list[ScenarioField] = []
            for field in raw.get("fields", []):
                raw_kind = str(field["kind"])
                if raw_kind not in ("boolean", "choice", "integer", "text"):
                    raise ValueError(f"Unsupported scenario field kind: {raw_kind}")
                fields.append(
                    ScenarioField(
                        name=str(field["name"]),
                        question=str(field["question"]),
                        kind=raw_kind,  # type: ignore[arg-type]
                        required=bool(field.get("required", False)),
                        options=tuple(str(item) for item in field.get("options", [])),
                        signals=tuple(str(item) for item in field.get("signals", [])),
                        minimum=field.get("minimum"),
                        maximum=field.get("maximum"),
                        required_when_field=field.get("required_when_field"),
                        required_when_equals=field.get("required_when_equals"),
                    )
                )
            equipment = tuple(
                ScenarioEquipment(
                    role=str(item["role"]),
                    keyword=str(item["keyword"]),
                    quantity=int(item.get("quantity", 1)),
                    quantity_field=item.get("quantity_field"),
                    include_when_field=item.get("include_when_field"),
                    include_when_equals=item.get("include_when_equals"),
                )
                for item in raw.get("equipment", [])
            )
            definitions.append(
                ScenarioDefinition(
                    scenario_id=str(raw["id"]),
                    name=str(raw["name"]),
                    aliases=tuple(str(item) for item in raw.get("aliases", [])),
                    fields=tuple(fields),
                    equipment=equipment,
                )
            )
        return cls(
            tuple(definitions),
            tuple(str(item) for item in payload.get("common_followup_signals", [])),
        )

    def definition(self, scenario_id: str) -> ScenarioDefinition | None:
        return self._definitions.get(scenario_id)

    def detect_scenario(self, message: str) -> ScenarioDefinition | None:
        normalized = message.casefold()
        matches = [
            (len(alias), definition)
            for definition in self._definitions.values()
            for alias in definition.aliases
            if alias.casefold() in normalized
        ]
        return max(matches, key=lambda item: item[0])[1] if matches else None

    def has_followup_signal(self, scenario_id: str, message: str) -> bool:
        definition = self.definition(scenario_id)
        if definition is None:
            return False
        normalized = message.casefold()
        return any(
            signal.casefold() in normalized for signal in self._common_followup_signals
        ) or any(
            signal.casefold() in normalized
            for field in definition.fields
            for signal in field.signals
        )

    def extraction_context(self) -> str:
        return json.dumps(
            [definition.extraction_schema() for definition in self._definitions.values()],
            ensure_ascii=False,
        )

    def build_plan(self, requirements: RentalRequirements) -> ScenarioPlan:
        definition = self.definition(requirements.scenario_id)
        if definition is None:
            raise ValueError(f"Unknown scenario: {requirements.scenario_id}")
        missing = [
            field
            for field in definition.fields
            if field.is_required(requirements.answers)
            and not field.valid(requirements.answers.get(field.name))
        ]
        needs: list[EquipmentNeed] = []
        if not missing:
            needs = [
                EquipmentNeed(
                    role=item.role,
                    keyword=item.keyword,
                    quantity=item.resolved_quantity(requirements.answers),
                )
                for item in definition.equipment
                if item.included(requirements.answers)
            ]
        clarification = None
        if missing:
            clarification = (
                f"为了给你组成完整的{definition.name}方案，请确认："
                + "；".join(field.question for field in missing)
                + "？"
            )
        return ScenarioPlan(
            requirements=requirements,
            scenario_name=definition.name,
            equipment_needs=tuple(needs),
            missing_fields=tuple(field.name for field in missing),
            clarification=clarification,
        )


def _merge_requirements(
    previous: RentalRequirements | None,
    extracted: RentalRequirements,
) -> RentalRequirements:
    if previous is None or previous.scenario_id != extracted.scenario_id:
        return extracted
    answers = {**previous.answers, **extracted.answers}
    return RentalRequirements(
        scenario_id=extracted.scenario_id,
        daily_budget=extracted.daily_budget or previous.daily_budget,
        answers=answers,
    )


class RentalRequirementsResolver:
    def __init__(self, catalog: ScenarioCatalog) -> None:
        self._catalog = catalog

    async def resolve(
        self,
        *,
        message: str,
        history: tuple[ModelMessage, ...],
        current: RentalRequirements | None,
        model: ChatModelPort,
        max_output_tokens: int,
    ) -> RequirementsResolution:
        recent_history = [item for item in history if item.role in ("user", "assistant")][-6:]
        if (
            not recent_history
            or recent_history[-1].role != "user"
            or recent_history[-1].content != message
        ):
            recent_history.append(ModelMessage(role="user", content=message))
        current_json = current.model_dump_json(by_alias=True) if current is not None else "null"
        prompt = f"""你只负责把设备租赁场景提取成结构化需求。
可用场景及字段: {self._catalog.extraction_context()}
当前已记住的需求: {current_json}

规则:
1. scenarioId 必须来自可用场景。
2. answers 的键必须来自该场景字段；只保存用户明确表达的值，不得根据常识替用户决定。
3. 否定表达必须保存为 boolean false，数量保存为 integer。
4. 每次都调用 {REQUIREMENTS_TOOL_NAME}，返回本轮能够确认的字段。
5. 不要决定设备清单、搜索商品、查询库存、计算价格或回答其他问题。"""
        request = ModelRequest(
            messages=(ModelMessage(role="system", content=prompt), *recent_history),
            tools=(
                ModelToolDefinition(
                    name=REQUIREMENTS_TOOL_NAME,
                    description="保存用户明确表达的结构化设备租赁需求。",
                    parameters=RentalRequirements.model_json_schema(by_alias=True),
                ),
            ),
            max_output_tokens=max_output_tokens,
            temperature=0.0,
            tool_choice=REQUIREMENTS_TOOL_NAME,
            enable_thinking=False,
            workload="action",
        )
        response = await model.complete(request)
        for call in response.tool_calls:
            if call.name != REQUIREMENTS_TOOL_NAME:
                continue
            try:
                extracted = RentalRequirements.model_validate(call.arguments)
                requirements = _merge_requirements(current, extracted)
                plan = self._catalog.build_plan(requirements)
            except (ValidationError, ValueError):
                return RequirementsResolution(
                    plan=None,
                    clarification="请重新说明使用场景、设备要求和预算。",
                    usage=response.usage,
                )
            return RequirementsResolution(
                plan=plan,
                clarification=plan.clarification,
                usage=response.usage,
            )
        return RequirementsResolution(
            plan=None,
            clarification=response.text.strip() or "请说明你的设备使用场景和预算。",
            usage=response.usage,
        )
