from decimal import Decimal
from pathlib import Path

from gearmate.config import Settings
from gearmate.llm.types import (
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
)
from gearmate.requirements import (
    RentalRequirements,
    RentalRequirementsResolver,
    ScenarioCatalog,
)


class FakeModel:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.response

    async def close(self) -> None:
        return None


def test_vague_scenario_request_requires_configured_fields() -> None:
    catalog = ScenarioCatalog.load_default()
    plan = catalog.build_plan(
        RentalRequirements(scenario_id="live_streaming", daily_budget=Decimal("500"))
    )

    assert not plan.ready
    assert plan.equipment_needs == ()
    assert plan.missing_fields == (
        "streaming_mode",
        "needs_audio",
        "needs_lighting",
    )
    assert plan.clarification is not None
    assert "手机直播还是相机直播" in plan.clarification


def test_complete_scenario_expands_equipment_from_catalog() -> None:
    catalog = ScenarioCatalog.load_default()
    plan = catalog.build_plan(
        RentalRequirements(
            scenario_id="live_streaming",
            daily_budget=Decimal("500"),
            answers={
                "streaming_mode": "camera",
                "camera_count": 1,
                "needs_audio": True,
                "needs_lighting": True,
            },
        )
    )

    assert plan.ready
    assert [item.role for item in plan.equipment_needs] == [
        "camera",
        "lens",
        "capture_card",
        "tripod",
        "microphone",
        "lighting",
    ]


async def test_resolver_merges_followup_with_remembered_requirements() -> None:
    catalog = ScenarioCatalog.load_default()
    model = FakeModel(
        ModelResponse(
            text="",
            finish_reason="tool_calls",
            usage=ModelUsage(input_tokens=45, output_tokens=9),
            tool_calls=(
                ModelToolCall(
                    id="requirements-1",
                    name="set_rental_requirements",
                    arguments={
                        "scenarioId": "live_streaming",
                        "answers": {
                            "streaming_mode": "camera",
                            "camera_count": 1,
                            "needs_audio": True,
                            "needs_lighting": False,
                        },
                    },
                ),
            ),
        )
    )
    resolver = RentalRequirementsResolver(catalog)

    result = await resolver.resolve(
        message="相机直播，一个机位，要麦克风，不用灯光",
        history=(),
        current=RentalRequirements(scenario_id="live_streaming", daily_budget=Decimal("500")),
        model=model,
        max_output_tokens=256,
    )

    assert result.plan is not None
    assert result.plan.ready
    assert result.plan.requirements.daily_budget == Decimal("500")
    assert result.plan.requirements.answers["needs_lighting"] is False
    assert [item.role for item in result.plan.equipment_needs] == [
        "camera",
        "lens",
        "capture_card",
        "tripod",
        "microphone",
    ]


def test_catalog_detection_does_not_treat_plain_product_as_scenario() -> None:
    catalog = ScenarioCatalog.load_default()

    assert catalog.detect_scenario("我需要直播设备") is not None
    assert catalog.detect_scenario("帮我找一个麦克风") is None
    assert catalog.has_followup_signal("live_streaming", "预算改成每天 600 元")


def test_default_roles_cover_configured_business_meeting_equipment() -> None:
    catalog = ScenarioCatalog.load_default()
    plan = catalog.build_plan(
        RentalRequirements(
            scenario_id="business_meeting",
            answers={
                "attendee_count": 20,
                "needs_projection": True,
                "needs_audio": True,
            },
        )
    )

    configured_roles = set(Settings(_env_file=None).equipment_roles)

    assert plan.ready
    assert {item.role for item in plan.equipment_needs} <= configured_roles


def test_new_scenario_is_added_by_configuration_only(tmp_path: Path) -> None:
    path = tmp_path / "scenarios.toml"
    path.write_text(
        """
[[scenarios]]
id = "podcast"
name = "播客录制设备"
aliases = ["播客"]

[[scenarios.fields]]
name = "guest_count"
question = "有几位嘉宾"
kind = "integer"
required = true
minimum = 1
maximum = 10
signals = ["嘉宾", "人"]

[[scenarios.equipment]]
role = "microphone"
keyword = "麦克风"
quantity_field = "guest_count"
""",
        encoding="utf-8",
    )
    catalog = ScenarioCatalog.from_toml(path)
    detected = catalog.detect_scenario("我要录制三人播客")
    assert detected is not None

    plan = catalog.build_plan(RentalRequirements(scenario_id="podcast", answers={"guest_count": 3}))

    assert plan.ready
    assert plan.equipment_needs[0].role == "microphone"
    assert plan.equipment_needs[0].quantity == 3
