import asyncio
import logging
from time import monotonic
from typing import Any

import httpx

from gearmate.actions import (
    AgentActionResolver,
    PendingProductSearch,
    PendingRentalAction,
    merge_pending_product_search,
    merge_pending_rental_action,
)
from gearmate.agent.graph import GearMateAgent
from gearmate.catalog import CatalogSearchService
from gearmate.config import Settings
from gearmate.llm.factory import build_chat_model
from gearmate.llm.openai_compatible import ModelConfigurationError
from gearmate.llm.port import ChatModelPort
from gearmate.llm.types import ModelUsage
from gearmate.memory import ConversationMemoryService
from gearmate.persistence.repositories import AgentRepository
from gearmate.prompts.loader import RenderedPrompt
from gearmate.recommendations import RecommendationPlanner
from gearmate.rental_period import (
    RentalPeriodPolicy,
    RentalPeriodResolver,
    has_temporal_signal,
)
from gearmate.rentflow.client import RentFlowClient
from gearmate.requirements import (
    RentalRequirementsResolver,
    ScenarioCatalog,
    ScenarioPlan,
)
from gearmate.search import RecentProductSearch
from gearmate.tools.contracts import RentalPeriodInput
from gearmate.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class RunCoordinator:
    def __init__(
        self,
        settings: Settings,
        repository: AgentRepository,
        rentflow_http: httpx.AsyncClient,
        prompt: RenderedPrompt,
        catalog_search: CatalogSearchService | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._rentflow_http = rentflow_http
        self._prompt = prompt
        self._catalog_search = catalog_search
        self._model: ChatModelPort | None = None
        self._model_lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._memory = ConversationMemoryService(repository, settings)
        rental_period_policy = RentalPeriodPolicy(settings.rental_period_max_advance_days)
        self._rental_period_resolver = RentalPeriodResolver(rental_period_policy)
        self._action_resolver = AgentActionResolver(settings.equipment_roles)
        self._scenario_catalog = ScenarioCatalog.load_default()
        self._requirements_resolver = RentalRequirementsResolver(self._scenario_catalog)

    async def start(
        self,
        *,
        conversation_id: str,
        user_id: str,
        access_token: str,
        message: str,
        rental_period: RentalPeriodInput | None,
    ) -> str:
        await self._repository.require_conversation(conversation_id, user_id)
        if rental_period is not None:
            await self._memory.remember_rental_period(conversation_id, rental_period)
        run = await self._repository.create_run(
            conversation_id,
            model_provider=self._settings.model_provider,
            model_id=self._settings.model_id,
            prompt_version=self._prompt.version,
            prompt_hash=self._prompt.content_hash,
            initial_state={"userMessage": message},
            user_message=message,
        )
        task = asyncio.create_task(
            self._execute(
                run_id=run.id,
                conversation_id=conversation_id,
                access_token=access_token,
                message=message,
                rental_period=rental_period,
            ),
            name=f"gearmate-run-{run.id}",
        )
        self._tasks[run.id] = task
        task.add_done_callback(lambda _: self._tasks.pop(run.id, None))
        return run.id

    async def cancel(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        if task is None or task.done():
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    async def close(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        if self._model is not None:
            await self._model.close()

    async def _model_client(self) -> ChatModelPort:
        if self._model is not None:
            return self._model
        async with self._model_lock:
            if self._model is None:
                self._model = build_chat_model(self._settings)
        return self._model

    async def _execute(
        self,
        *,
        run_id: str,
        conversation_id: str,
        access_token: str,
        message: str,
        rental_period: RentalPeriodInput | None,
    ) -> None:
        async def write_event(event_type: str, payload: dict[str, Any]) -> None:
            await self._repository.append_event(run_id, event_type, payload)

        try:
            model = await self._model_client()
            started = monotonic()
            context = await self._memory.build_context(conversation_id)
            effective_rental_period = rental_period or context.rental_period
            action_resolution = await self._action_resolver.resolve(
                message=message,
                history=context.messages,
                current_scenario_id=(
                    context.rental_requirements.scenario_id
                    if context.rental_requirements is not None
                    else None
                ),
                pending_product_search=context.pending_product_search,
                pending_rental_action=context.pending_rental_action,
                model=model,
                max_output_tokens=(self._settings.action_resolution_max_output_tokens),
                recent_product_search_json=(
                    context.recent_product_search.model_dump_json(by_alias=True)
                    if context.recent_product_search is not None
                    else "none"
                ),
                recent_product_ids=(
                    tuple(
                        item.product_id for item in context.recent_product_search.items
                    )
                    if context.recent_product_search is not None
                    else ()
                ),
                catalog_vocabulary=(
                    await self._catalog_search.vocabulary()
                    if self._catalog_search is not None
                    else None
                ),
            )
            action_usage = action_resolution.usage
            action_rounds = 1
            if action_resolution.action is None:
                await self._complete_clarification(
                    run_id=run_id,
                    text=action_resolution.clarification or "请明确本轮希望执行的操作。",
                    usage=action_usage,
                    duration_ms=round((monotonic() - started) * 1000),
                    model_rounds=action_rounds,
                )
                return
            action = action_resolution.action
            action = merge_pending_product_search(
                action,
                context.pending_product_search,
            )
            action = merge_pending_rental_action(
                action,
                context.pending_rental_action,
            )
            if (
                action.action not in ("chat", "product_search")
                and context.pending_product_search is not None
            ):
                await self._memory.clear_pending_product_search(conversation_id)
            if (
                action.action not in ("chat", "availability", "quote")
                and context.pending_rental_action is not None
            ):
                await self._memory.clear_pending_rental_action(conversation_id)

            pending_product_search: PendingProductSearch | None = None
            if action.action == "product_search":
                waiting_for_period = rental_period is None and (
                    has_temporal_signal(message)
                    or (
                        action.continues_pending
                        and context.pending_product_search is not None
                        and context.pending_product_search.waiting_for_rental_period
                    )
                )
                pending_product_search = PendingProductSearch.from_action(
                    action,
                    waiting_for_rental_period=waiting_for_period,
                )
                await self._memory.remember_pending_product_search(
                    conversation_id,
                    pending_product_search,
                )
            pending_rental_action: PendingRentalAction | None = None
            if (
                action.action in ("availability", "quote")
                and action.product_id is not None
                and (
                    effective_rental_period is None
                    or has_temporal_signal(message)
                    or (action.continues_pending and context.pending_rental_action is not None)
                )
            ):
                pending_rental_action = PendingRentalAction.from_action(action)
                await self._memory.remember_pending_rental_action(
                    conversation_id,
                    pending_rental_action,
                )
            await write_event(
                "action.resolved",
                action.model_dump(mode="json", by_alias=True),
            )

            resolver_usage = ModelUsage()
            resolver_rounds = 0
            should_resolve_period = rental_period is None and (
                has_temporal_signal(message)
                or (
                    pending_product_search is not None
                    and pending_product_search.waiting_for_rental_period
                )
                or (pending_rental_action is not None and action.continues_pending)
            )
            if should_resolve_period:
                resolution = await self._rental_period_resolver.resolve(
                    message=message,
                    timezone=context.timezone,
                    now_utc=context.now_utc,
                    now_local=context.now_local,
                    history=context.messages,
                    model=model,
                    max_output_tokens=(self._settings.rental_period_extraction_max_output_tokens),
                )
                resolver_usage = resolution.usage
                resolver_rounds = 1
                if resolution.rental_period is not None:
                    effective_rental_period = resolution.rental_period
                    await self._memory.remember_rental_period(
                        conversation_id,
                        resolution.rental_period,
                        now_utc=context.now_utc,
                    )
                    if pending_product_search is not None:
                        pending_product_search = pending_product_search.model_copy(
                            update={"waiting_for_rental_period": False}
                        )
                        await self._memory.remember_pending_product_search(
                            conversation_id,
                            pending_product_search,
                        )
                else:
                    await self._complete_clarification(
                        run_id=run_id,
                        text=resolution.clarification or "请确认完整的开始和结束时间。",
                        usage=self._combine_usage(action_usage, resolver_usage),
                        duration_ms=round((monotonic() - started) * 1000),
                        model_rounds=action_rounds + resolver_rounds,
                    )
                    return
            requirements_usage = ModelUsage()
            requirements_rounds = 0
            scenario_plan: ScenarioPlan | None = None
            should_resolve_requirements = action.action == "scenario_continue"
            if should_resolve_requirements and context.rental_requirements is not None:
                scenario_plan = self._scenario_catalog.build_plan(context.rental_requirements)
            if should_resolve_requirements:
                requirements_resolution = await self._requirements_resolver.resolve(
                    message=message,
                    history=context.messages,
                    current=context.rental_requirements,
                    model=model,
                    max_output_tokens=(self._settings.requirements_extraction_max_output_tokens),
                )
                requirements_usage = requirements_resolution.usage
                requirements_rounds = 1
                scenario_plan = requirements_resolution.plan
                if scenario_plan is not None:
                    await self._memory.remember_requirements(
                        conversation_id, scenario_plan.requirements
                    )
                    await write_event(
                        "requirements.resolved",
                        {
                            **scenario_plan.model_context(),
                            "ready": scenario_plan.ready,
                            "missingFields": list(scenario_plan.missing_fields),
                        },
                    )
                if scenario_plan is None or not scenario_plan.ready:
                    clarification_usage = self._combine_usage(
                        resolver_usage,
                        action_usage,
                        requirements_usage,
                    )
                    await self._complete_clarification(
                        run_id=run_id,
                        text=requirements_resolution.clarification or "请补充完整的设备租赁需求。",
                        usage=clarification_usage,
                        duration_ms=round((monotonic() - started) * 1000),
                        model_rounds=(resolver_rounds + action_rounds + requirements_rounds),
                    )
                    return
            tools = ToolRegistry(
                RentFlowClient(self._rentflow_http, access_token),
                timeout_seconds=self._settings.tool_timeout_seconds,
                max_result_items=self._settings.max_tool_result_items,
                max_concurrency=self._settings.max_tool_concurrency,
                scenario_plan=scenario_plan,
                catalog_search=self._catalog_search,
            )
            agent = GearMateAgent(model, tools, self._settings, self._prompt)
            async with asyncio.timeout(self._settings.run_timeout_seconds):
                result = await agent.run(
                    message=message,
                    history=list(context.messages),
                    rental_period=effective_rental_period,
                    scenario_plan=scenario_plan,
                    action=action,
                    write_event=write_event,
                )
            if tools.last_search_diagnostics is not None:
                await write_event("search.retrieval", tools.last_search_diagnostics)
            presentation_payload: dict[str, Any] | None = None
            if action.action == "product_search" and tools.last_product_search_result is not None:
                presentation = RecommendationPlanner().plan(
                    tools.last_product_search_result,
                    action,
                    effective_rental_period,
                )
                presentation_payload = presentation.model_dump(mode="json", by_alias=True)
                await write_event("recommendation.presented", presentation_payload)
                await self._memory.remember_recent_product_search(
                    conversation_id,
                    RecentProductSearch.from_result(tools.last_product_search_result),
                )
            elif (
                action.action in ("availability", "quote")
                and action.product_id is not None
                and effective_rental_period is not None
                and context.recent_product_search is not None
                and (
                    tools.last_availability_result is not None
                    or tools.last_quote_result is not None
                )
            ):
                product = next(
                    (
                        item
                        for item in context.recent_product_search.items
                        if item.product_id == action.product_id
                    ),
                    None,
                )
                if product is not None:
                    exact_presentation = RecommendationPlanner().plan_exact(
                        product,
                        effective_rental_period,
                        (
                            tools.last_availability_result.available_count
                            if tools.last_availability_result is not None
                            else None
                        ),
                    )
                    if exact_presentation is not None:
                        presentation_payload = exact_presentation.model_dump(
                            mode="json", by_alias=True
                        )
                        await write_event("recommendation.presented", presentation_payload)
            if action.action == "product_search" and result.tool_call_count > 0:
                follow_up = (
                    presentation_payload.get("followUp")
                    if presentation_payload is not None
                    else None
                )
                if isinstance(follow_up, dict) and follow_up.get("field") in {
                    "use_case",
                    "rental_period",
                }:
                    if (
                        follow_up.get("field") == "rental_period"
                        and pending_product_search is not None
                    ):
                        pending_product_search = pending_product_search.model_copy(
                            update={"waiting_for_rental_period": True}
                        )
                        await self._memory.remember_pending_product_search(
                            conversation_id,
                            pending_product_search,
                        )
                else:
                    await self._memory.clear_pending_product_search(conversation_id)
            if action.action in ("availability", "quote") and result.tool_call_count > 0:
                await self._memory.clear_pending_rental_action(conversation_id)
            preprocessing_usage = self._combine_usage(
                resolver_usage, action_usage, requirements_usage
            )
            input_tokens = result.input_tokens + preprocessing_usage.input_tokens
            output_tokens = result.output_tokens + preprocessing_usage.output_tokens
            model_rounds = (
                result.model_rounds + resolver_rounds + requirements_rounds + action_rounds
            )
            state = {
                "reply": result.text,
                "stopReason": result.stop_reason,
                "modelRounds": model_rounds,
                "toolCallCount": result.tool_call_count,
                "durationMs": round((monotonic() - started) * 1000),
                **({"presentation": presentation_payload} if presentation_payload else {}),
            }
            await self._repository.finalize_run(
                run_id,
                event_type="run.completed",
                event_payload=state,
                status="COMPLETED",
                stop_reason=result.stop_reason,
                error_code=result.error_code,
                state=state,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_rounds=model_rounds,
                tool_call_count=result.tool_call_count,
            )
            try:
                await self._memory.maybe_summarize(conversation_id, model)
            except Exception:
                logger.exception(
                    "Conversation summarization failed (conversation_id=%s)",
                    conversation_id,
                )
        except TimeoutError:
            await self._fail(run_id, "TIMEOUT", "RUN_TIMEOUT")
        except ModelConfigurationError:
            await self._fail(run_id, "MODEL_CONFIGURATION_ERROR", "MODEL_CONFIGURATION_ERROR")
        except asyncio.CancelledError:
            await self._fail(run_id, "CANCELLED", "RUN_CANCELLED", status="CANCELLED")
            raise
        except Exception:
            logger.exception("Agent run failed (run_id=%s)", run_id)
            await self._fail(run_id, "FAILED", "AGENT_RUN_FAILED")

    async def _complete_clarification(
        self,
        *,
        run_id: str,
        text: str,
        usage: ModelUsage,
        duration_ms: int,
        model_rounds: int,
    ) -> None:
        await self._repository.append_event(run_id, "assistant.delta", {"content": text})
        await self._repository.append_event(
            run_id,
            "assistant.completed",
            {"content": text, "stopReason": "NEED_CLARIFICATION"},
        )
        state = {
            "reply": text,
            "stopReason": "NEED_CLARIFICATION",
            "modelRounds": model_rounds,
            "toolCallCount": 0,
            "durationMs": duration_ms,
        }
        await self._repository.finalize_run(
            run_id,
            event_type="run.completed",
            event_payload=state,
            status="COMPLETED",
            stop_reason="NEED_CLARIFICATION",
            error_code=None,
            state=state,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            model_rounds=model_rounds,
            tool_call_count=0,
        )

    @staticmethod
    def _combine_usage(*items: ModelUsage) -> ModelUsage:
        return ModelUsage(
            input_tokens=sum(item.input_tokens for item in items),
            output_tokens=sum(item.output_tokens for item in items),
        )

    async def _fail(
        self,
        run_id: str,
        stop_reason: str,
        error_code: str,
        *,
        status: str = "FAILED",
    ) -> None:
        payload = {"stopReason": stop_reason, "errorCode": error_code}
        try:
            await self._repository.finalize_run(
                run_id,
                event_type="run.failed",
                event_payload=payload,
                status=status,
                stop_reason=stop_reason,
                error_code=error_code,
                state=payload,
                input_tokens=0,
                output_tokens=0,
                model_rounds=0,
                tool_call_count=0,
            )
        except Exception:
            logger.exception("Failed to persist terminal run state (run_id=%s)", run_id)
            # The task cannot repair a database outage; startup reconciliation can close stale runs.
            return
