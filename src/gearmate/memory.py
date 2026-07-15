from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from typing import Literal, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from gearmate.actions import PendingProductSearch, PendingRentalAction
from gearmate.config import Settings
from gearmate.llm.port import ChatModelPort
from gearmate.llm.types import ModelMessage, ModelRequest, ModelUsage
from gearmate.rental_period import InvalidRentalPeriod, RentalPeriodPolicy
from gearmate.requirements import RentalRequirements
from gearmate.tools.contracts import RentalPeriodInput

SUMMARY_SYSTEM_PROMPT = """你负责压缩 GearMate 的较早会话历史。
只保留用户目标、偏好、已确认租期、数量、预算、用途、候选商品和未解决问题。
不要把库存数量、价格或报价描述成当前仍然有效的事实; 这些信息必须在新一轮中重新查询。
不要添加原文中不存在的信息。使用简洁中文纯文本, 不要输出 JSON 或 Markdown 标题。"""

SUMMARY_CONTEXT_PREFIX = """以下是更早会话的摘要, 仅用于理解用户目标和已确认约束。
摘要中的库存、价格和报价都可能已经过期, 引用前必须重新调用工具:
"""

TIME_CONTEXT_TEMPLATE = """当前可信时间:
- UTC: {now_utc}
- 用户时区: {timezone}
- 用户当地时间: {now_local}
“今天”“明天”“下周五”等相对日期必须以该时间为基准。
缺少明确开始时间、结束时间或时区时必须要求用户确认, 不得猜测。"""


@dataclass(frozen=True, slots=True)
class ConversationStateMemory:
    rental_start_at: datetime | None
    rental_end_at: datetime | None
    rental_requirements: RentalRequirements | None = None
    pending_product_search: PendingProductSearch | None = None
    pending_rental_action: PendingRentalAction | None = None


@dataclass(frozen=True, slots=True)
class ConversationSummaryMemory:
    content: str
    through_event_id: str
    source_message_count: int
    estimated_tokens: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ConversationMessageMemory:
    event_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class MemoryRepository(Protocol):
    async def conversation_timezone(self, conversation_id: str) -> str: ...

    async def upsert_conversation_rental_period(
        self, conversation_id: str, rental_period: RentalPeriodInput
    ) -> None: ...

    async def clear_conversation_rental_period(self, conversation_id: str) -> None: ...

    async def upsert_pending_product_search(
        self,
        conversation_id: str,
        pending_search: PendingProductSearch,
    ) -> None: ...

    async def clear_pending_product_search(self, conversation_id: str) -> None: ...

    async def upsert_pending_rental_action(
        self,
        conversation_id: str,
        pending_action: PendingRentalAction,
    ) -> None: ...

    async def clear_pending_rental_action(self, conversation_id: str) -> None: ...

    async def upsert_conversation_requirements(
        self, conversation_id: str, requirements: RentalRequirements
    ) -> None: ...

    async def conversation_state(self, conversation_id: str) -> ConversationStateMemory | None: ...

    async def latest_conversation_summary(
        self, conversation_id: str
    ) -> ConversationSummaryMemory | None: ...

    async def recent_conversation_messages(
        self,
        conversation_id: str,
        limit: int,
        after_event_id: str | None = None,
    ) -> list[ConversationMessageMemory]: ...

    async def conversation_messages_after(
        self, conversation_id: str, after_event_id: str | None, limit: int
    ) -> list[ConversationMessageMemory]: ...

    async def save_conversation_summary(
        self,
        conversation_id: str,
        content: str,
        through_event_id: str,
        source_message_count: int,
        estimated_tokens: int,
        input_tokens: int,
        output_tokens: int,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ConversationContext:
    messages: tuple[ModelMessage, ...]
    rental_period: RentalPeriodInput | None
    rental_requirements: RentalRequirements | None
    pending_product_search: PendingProductSearch | None
    pending_rental_action: PendingRentalAction | None
    timezone: str
    now_utc: datetime
    now_local: datetime


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, ceil(len(text) / 4), ceil(len(text.encode("utf-8")) / 3))


class ConversationMemoryService:
    def __init__(self, repository: MemoryRepository, settings: Settings) -> None:
        self._repository = repository
        self._settings = settings
        self._rental_period_policy = RentalPeriodPolicy(settings.rental_period_max_advance_days)

    async def remember_rental_period(
        self,
        conversation_id: str,
        rental_period: RentalPeriodInput,
        *,
        now_utc: datetime | None = None,
    ) -> None:
        self._rental_period_policy.validate(rental_period, now_utc=now_utc)
        await self._repository.upsert_conversation_rental_period(conversation_id, rental_period)

    async def remember_requirements(
        self, conversation_id: str, requirements: RentalRequirements
    ) -> None:
        await self._repository.upsert_conversation_requirements(conversation_id, requirements)

    async def remember_pending_product_search(
        self,
        conversation_id: str,
        pending_search: PendingProductSearch,
    ) -> None:
        await self._repository.upsert_pending_product_search(
            conversation_id,
            pending_search,
        )

    async def clear_pending_product_search(self, conversation_id: str) -> None:
        await self._repository.clear_pending_product_search(conversation_id)

    async def remember_pending_rental_action(
        self,
        conversation_id: str,
        pending_action: PendingRentalAction,
    ) -> None:
        await self._repository.upsert_pending_rental_action(
            conversation_id,
            pending_action,
        )

    async def clear_pending_rental_action(self, conversation_id: str) -> None:
        await self._repository.clear_pending_rental_action(conversation_id)

    async def build_context(
        self, conversation_id: str, now_utc: datetime | None = None
    ) -> ConversationContext:
        timezone = await self._repository.conversation_timezone(conversation_id)
        reference_utc = now_utc or datetime.now(UTC)
        if reference_utc.tzinfo is None:
            reference_utc = reference_utc.replace(tzinfo=UTC)
        reference_utc = reference_utc.astimezone(UTC)
        state = await self._repository.conversation_state(conversation_id)
        summary = await self._repository.latest_conversation_summary(conversation_id)
        recent = await self._repository.recent_conversation_messages(
            conversation_id,
            self._settings.context_source_message_limit,
            summary.through_event_id if summary is not None else None,
        )
        try:
            zone = ZoneInfo(timezone)
        except ZoneInfoNotFoundError:
            timezone = "UTC"
            zone = ZoneInfo("UTC")
        reference_local = reference_utc.astimezone(zone)
        time_content = TIME_CONTEXT_TEMPLATE.format(
            now_utc=reference_utc.isoformat(),
            timezone=timezone,
            now_local=reference_local.isoformat(),
        )
        messages: list[ModelMessage] = [ModelMessage(role="system", content=time_content)]
        used_tokens = estimate_tokens(time_content)
        if summary is not None:
            summary_content = SUMMARY_CONTEXT_PREFIX + summary.content
            messages.append(ModelMessage(role="system", content=summary_content))
            used_tokens += estimate_tokens(summary_content)

        selected: list[ConversationMessageMemory] = []
        for item in reversed(recent):
            item_tokens = estimate_tokens(item.content) + 8
            if selected and (
                used_tokens + item_tokens > self._settings.context_history_token_budget
            ):
                break
            selected.append(item)
            used_tokens += item_tokens
        messages.extend(
            ModelMessage(role=item.role, content=item.content) for item in reversed(selected)
        )

        rental_period = None
        if (
            state is not None
            and state.rental_start_at is not None
            and state.rental_end_at is not None
        ):
            stored_period = RentalPeriodInput(
                start_at=state.rental_start_at,
                end_at=state.rental_end_at,
            )
            try:
                rental_period = self._rental_period_policy.validate(
                    stored_period, now_utc=reference_utc
                )
            except InvalidRentalPeriod:
                await self._repository.clear_conversation_rental_period(conversation_id)
        return ConversationContext(
            messages=tuple(messages),
            rental_period=rental_period,
            rental_requirements=(state.rental_requirements if state is not None else None),
            pending_product_search=(state.pending_product_search if state is not None else None),
            pending_rental_action=(state.pending_rental_action if state is not None else None),
            timezone=timezone,
            now_utc=reference_utc,
            now_local=reference_local,
        )

    async def maybe_summarize(
        self, conversation_id: str, model: ChatModelPort
    ) -> ModelUsage | None:
        previous = await self._repository.latest_conversation_summary(conversation_id)
        source = await self._repository.conversation_messages_after(
            conversation_id,
            previous.through_event_id if previous is not None else None,
            self._settings.context_source_message_limit,
        )
        total_tokens = sum(estimate_tokens(item.content) + 8 for item in source)
        keep_recent = self._settings.context_recent_messages
        if (
            total_tokens < self._settings.context_summary_trigger_tokens
            or len(source) <= keep_recent
        ):
            return None

        candidate_tokens = estimate_tokens(previous.content) if previous else 0
        candidates: list[ConversationMessageMemory] = []
        for item in source[:-keep_recent]:
            item_tokens = estimate_tokens(item.content) + 8
            if candidates and (
                candidate_tokens + item_tokens > self._settings.context_history_token_budget
            ):
                break
            candidates.append(item)
            candidate_tokens += item_tokens
        if not candidates:
            return None
        transcript = "\n".join(f"{item.role}: {item.content}" for item in candidates)
        previous_text = previous.content if previous is not None else "(无)"
        request = ModelRequest(
            messages=(
                ModelMessage(role="system", content=SUMMARY_SYSTEM_PROMPT),
                ModelMessage(
                    role="user",
                    content=(f"已有摘要:\n{previous_text}\n\n需要合并的新对话:\n{transcript}"),
                ),
            ),
            max_output_tokens=self._settings.context_summary_max_output_tokens,
            temperature=0.0,
        )
        response = await model.complete(request)
        content = response.text.strip()
        if not content:
            return response.usage
        await self._repository.save_conversation_summary(
            conversation_id=conversation_id,
            content=content,
            through_event_id=candidates[-1].event_id,
            source_message_count=len(candidates),
            estimated_tokens=estimate_tokens(content),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response.usage
