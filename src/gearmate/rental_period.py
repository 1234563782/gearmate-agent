import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from gearmate.llm.port import ChatModelPort
from gearmate.llm.types import (
    ModelMessage,
    ModelRequest,
    ModelToolDefinition,
    ModelUsage,
)
from gearmate.tools.contracts import RentalPeriodInput

TEMPORAL_SIGNAL = re.compile(
    r"(?:今天|明天|后天|大后天|本周|这周|下周|周末|星期[一二三四五六日天]?|"
    r"周[一二三四五六日天]|月底|月初|早上|上午|中午|下午|晚上|凌晨|"
    r"\d{4}\s*[年/-]\s*\d{1,2}|\d{1,2}\s*月\s*\d{1,2}\s*[日号]?|"
    r"\d{1,2}\s*[点时]|[零〇一二两三四五六七八九十百\d]+\s*(?:天|小时|周|个?月)|"
    r"today|tomorrow|tonight|"
    r"next\s+(?:week|monday|tuesday|wednesday|thursday|friday|saturday|sunday))",
    re.IGNORECASE,
)

RESOLVER_TOOL_NAME = "set_rental_period"


@dataclass(frozen=True, slots=True)
class RentalPeriodResolution:
    rental_period: RentalPeriodInput | None
    clarification: str | None
    usage: ModelUsage


def has_temporal_signal(message: str) -> bool:
    return TEMPORAL_SIGNAL.search(message) is not None


def resolver_system_prompt(timezone: str, now_utc: datetime, now_local: datetime) -> str:
    return f"""你只负责从租赁对话中解析明确的开始时间和结束时间。
可信 UTC 时间: {now_utc.isoformat()}
用户时区: {timezone}
用户当地时间: {now_local.isoformat()}

规则:
1. 相对日期必须以上述用户当地时间为基准。
2. 只有开始日期、开始时间、结束日期、结束时间都能唯一确定时, 才调用 {RESOLVER_TOOL_NAME}。
3. 输出时间必须包含用户时区对应的 UTC offset。
4. "上午"、"下午"、"晚上"、"周末"等没有具体时分的表达不算完整。
5. 信息不完整或有多种解释时不要调用工具, 只返回一个简洁的中文确认问题。
6. 不要搜索商品、查询库存、计算价格或回答其他问题。"""


class RentalPeriodResolver:
    async def resolve(
        self,
        *,
        message: str,
        timezone: str,
        now_utc: datetime,
        now_local: datetime,
        history: tuple[ModelMessage, ...],
        model: ChatModelPort,
        max_output_tokens: int,
    ) -> RentalPeriodResolution:
        recent_history = [
            item for item in history if item.role in ("user", "assistant")
        ][-6:]
        if (
            not recent_history
            or recent_history[-1].role != "user"
            or recent_history[-1].content != message
        ):
            recent_history.append(ModelMessage(role="user", content=message))
        request = ModelRequest(
            messages=(
                ModelMessage(
                    role="system",
                    content=resolver_system_prompt(
                        timezone, now_utc, now_local
                    ),
                ),
                *recent_history,
            ),
            tools=(
                ModelToolDefinition(
                    name=RESOLVER_TOOL_NAME,
                    description="保存用户已经明确确认的租赁开始和结束时间。",
                    parameters=RentalPeriodInput.model_json_schema(by_alias=True),
                ),
            ),
            max_output_tokens=max_output_tokens,
            temperature=0.0,
        )
        response = await model.complete(request)
        for call in response.tool_calls:
            if call.name != RESOLVER_TOOL_NAME:
                continue
            try:
                period = RentalPeriodInput.model_validate(call.arguments)
            except ValidationError:
                return RentalPeriodResolution(
                    rental_period=None,
                    clarification="请确认完整的开始日期时间和结束日期时间。",
                    usage=response.usage,
                )
            zone = ZoneInfo(timezone)
            if (
                period.start_at.utcoffset()
                != period.start_at.astimezone(zone).utcoffset()
                or period.end_at.utcoffset()
                != period.end_at.astimezone(zone).utcoffset()
            ):
                return RentalPeriodResolution(
                    rental_period=None,
                    clarification=f"请按 {timezone} 时区确认开始和结束时间。",
                    usage=response.usage,
                )
            if period.start_at <= now_utc:
                return RentalPeriodResolution(
                    rental_period=None,
                    clarification="租赁开始时间必须晚于当前时间, 请重新确认。",
                    usage=response.usage,
                )
            return RentalPeriodResolution(
                rental_period=period,
                clarification=None,
                usage=response.usage,
            )
        clarification = response.text.strip() or (
            "请确认具体的开始日期时间和结束日期时间。"
        )
        return RentalPeriodResolution(
            rental_period=None,
            clarification=clarification,
            usage=response.usage,
        )
