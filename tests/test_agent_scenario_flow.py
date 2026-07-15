from decimal import Decimal

from gearmate.actions import AgentAction
from gearmate.agent.graph import GearMateAgent
from gearmate.config import Settings
from gearmate.llm.types import ModelRequest, ModelResponse, ModelUsage
from gearmate.prompts.loader import RenderedPrompt
from gearmate.requirements import RentalRequirements, ScenarioCatalog
from gearmate.tools.contracts import ScenarioKitResult
from gearmate.tools.registry import ToolExecutionResult


class FakeModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            text="当前目录不能组成完整方案，预算为 500 元。",
            finish_reason="stop",
            usage=ModelUsage(input_tokens=30, output_tokens=8),
        )

    async def close(self) -> None:
        return None


class FakeTools:
    def __init__(self) -> None:
        self.called_names: list[str] = []

    def model_definitions(self):
        return ()

    async def execute_all(self, calls, facts, write_event):
        self.called_names.extend(call.name for call in calls)
        call = calls[0]
        result = ScenarioKitResult(
            scenario="live_streaming",
            items=(),
            total_daily_rate="0.00",
            max_daily_budget="500.00",
            within_budget=False,
            availability_checked=False,
            missing_roles=("camera",),
        )
        facts.add(result)
        await write_event("tool.completed", {"toolCallId": call.id, "tool": call.name})
        return [
            ToolExecutionResult(
                call=call,
                content=result.model_dump_json(by_alias=True),
                is_error=False,
                result=result,
            )
        ]


async def test_complete_budgeted_scenario_runs_kit_before_model() -> None:
    plan = ScenarioCatalog.load_default().build_plan(
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
    model = FakeModel()
    tools = FakeTools()
    events: list[str] = []

    async def write_event(event_type, payload):
        events.append(event_type)

    result = await GearMateAgent(
        model,
        tools,  # type: ignore[arg-type]
        Settings(_env_file=None),
        RenderedPrompt(version="test", content_hash="hash", content="system"),
    ).run(
        message="完整场景需求",
        history=[],
        rental_period=None,
        scenario_plan=plan,
        action=AgentAction(action="scenario_continue"),
        write_event=write_event,
    )

    assert tools.called_names == ["recommend_scenario_kit"]
    assert "model.started" not in events
    assert model.requests == []
    assert result.stop_reason == "COMPLETED"
