import asyncio
import logging
from time import monotonic
from typing import Any

import httpx

from gearmate.agent.graph import GearMateAgent
from gearmate.config import Settings
from gearmate.llm.factory import build_chat_model
from gearmate.llm.openai_compatible import ModelConfigurationError
from gearmate.llm.port import ChatModelPort
from gearmate.llm.types import ModelUsage
from gearmate.memory import ConversationMemoryService
from gearmate.persistence.repositories import AgentRepository
from gearmate.prompts.loader import RenderedPrompt
from gearmate.rental_period import RentalPeriodResolver, has_temporal_signal
from gearmate.rentflow.client import RentFlowClient
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
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._rentflow_http = rentflow_http
        self._prompt = prompt
        self._model: ChatModelPort | None = None
        self._model_lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._memory = ConversationMemoryService(repository, settings)
        self._rental_period_resolver = RentalPeriodResolver()

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
            await self._memory.remember_rental_period(
                conversation_id, rental_period
            )
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
            tools = ToolRegistry(
                RentFlowClient(self._rentflow_http, access_token),
                timeout_seconds=self._settings.tool_timeout_seconds,
                max_result_items=self._settings.max_tool_result_items,
                max_concurrency=self._settings.max_tool_concurrency,
            )
            context = await self._memory.build_context(conversation_id)
            effective_rental_period = rental_period or context.rental_period
            resolver_usage = ModelUsage()
            resolver_rounds = 0
            if rental_period is None and has_temporal_signal(message):
                resolution = await self._rental_period_resolver.resolve(
                    message=message,
                    timezone=context.timezone,
                    now_utc=context.now_utc,
                    now_local=context.now_local,
                    history=context.messages,
                    model=model,
                    max_output_tokens=(
                        self._settings.rental_period_extraction_max_output_tokens
                    ),
                )
                resolver_usage = resolution.usage
                resolver_rounds = 1
                if resolution.rental_period is not None:
                    effective_rental_period = resolution.rental_period
                    await self._memory.remember_rental_period(
                        conversation_id, resolution.rental_period
                    )
                else:
                    await self._complete_clarification(
                        run_id=run_id,
                        text=resolution.clarification
                        or "请确认完整的开始和结束时间。",
                        usage=resolver_usage,
                        duration_ms=round((monotonic() - started) * 1000),
                    )
                    return
            agent = GearMateAgent(model, tools, self._settings, self._prompt)
            async with asyncio.timeout(self._settings.run_timeout_seconds):
                result = await agent.run(
                    message=message,
                    history=list(context.messages),
                    rental_period=effective_rental_period,
                    write_event=write_event,
                )
            input_tokens = result.input_tokens + resolver_usage.input_tokens
            output_tokens = result.output_tokens + resolver_usage.output_tokens
            model_rounds = result.model_rounds + resolver_rounds
            state = {
                "reply": result.text,
                "stopReason": result.stop_reason,
                "modelRounds": model_rounds,
                "toolCallCount": result.tool_call_count,
                "durationMs": round((monotonic() - started) * 1000),
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
    ) -> None:
        await self._repository.append_event(
            run_id, "assistant.delta", {"content": text}
        )
        await self._repository.append_event(
            run_id,
            "assistant.completed",
            {"content": text, "stopReason": "NEED_CLARIFICATION"},
        )
        state = {
            "reply": text,
            "stopReason": "NEED_CLARIFICATION",
            "modelRounds": 1,
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
            model_rounds=1,
            tool_call_count=0,
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
