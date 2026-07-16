from gearmate.actions import AgentAction
from gearmate.tools.contracts import RentalPeriodInput
from gearmate.validation.facts import FactSnapshot


class UserResponseComposer:
    def compose(
        self,
        *,
        action: AgentAction,
        facts: FactSnapshot,
        rental_period: RentalPeriodInput | None,
    ) -> str:
        if facts.scenario_kits:
            return self._scenario(facts)
        if action.action == "product_search":
            return self._product_search(action, facts, rental_period)
        if action.action == "product_detail":
            return self._product_detail(facts)
        if action.action == "availability":
            return self._availability(facts)
        if action.action == "quote":
            return self._quote(facts)
        return facts.fallback_text()

    def _product_search(
        self,
        action: AgentAction,
        facts: FactSnapshot,
        rental_period: RentalPeriodInput | None,
    ) -> str:
        products = list(facts.products.values())
        if not products:
            return (
                "暂时没有找到符合这些条件的设备。"
                "可以放宽用途、价格或品牌要求，我再帮你重新筛选。"
            )
        if action.target_daily_rate is not None:
            intro = f"我按目标日租 ¥{action.target_daily_rate} 和你的使用需求筛选了这些设备："
        elif action.max_daily_rate is not None:
            intro = f"我按日租不超过 ¥{action.max_daily_rate} 和你的使用需求筛选了这些设备："
        else:
            intro = "我按你的设备类型和使用需求筛选了这些候选："

        lines = [intro]
        for index, product in enumerate(products[:4]):
            reasons = "、".join(item.name for item in product.use_cases[:2])
            detail = f"日租 ¥{product.daily_rate}"
            if reasons:
                detail += f"，适合{reasons}"
            if product.available_count is not None:
                detail += f"，当前租期可租 {product.available_count} 台"
            prefix = "优先推荐" if index == 0 else "备选"
            lines.append(f"- {prefix} {product.name}：{detail}。")

        if rental_period is not None:
            lines.append("我已经按你给出的租期核验库存，点开卡片即可查看完整报价并预订。")
        else:
            lines.append("目前显示的是日租参考价；选定租期后，我可以继续查询实时库存和总价。")
        if len(products) > 1:
            lines.append("还可以继续告诉我你更看重性能、便携性还是预算，我会进一步缩小范围。")
        return "\n".join(lines)

    @staticmethod
    def _product_detail(facts: FactSnapshot) -> str:
        product = next(iter(facts.products.values()), None)
        if product is None:
            return "暂时没有取得这款设备的详细信息，请重新选择商品。"
        reasons = "、".join(item.name for item in product.use_cases[:3])
        lines = [
            f"这款是 {product.name}（{product.brand} {product.model}）。",
            f"日租 ¥{product.daily_rate}，固定押金 ¥{product.fixed_deposit}。",
        ]
        if reasons:
            lines.append(f"目录中标注的主要适用场景是：{reasons}。")
        lines.append("告诉我具体租期后，我可以继续核验库存和完整报价。")
        return "\n".join(lines)

    @staticmethod
    def _availability(facts: FactSnapshot) -> str:
        availability = next(iter(facts.availability.values()), None)
        if availability is None:
            return "暂时没有取得这个租期的实时库存，请稍后重试。"
        if availability.available:
            return (
                f"这个租期可以租，目前还有 {availability.available_count} 台。\n"
                "点开下方卡片可以查看完整报价并继续预订。"
            )
        return "这个租期暂时没有可租库存。可以调整时间，我再帮你重新查询。"

    @staticmethod
    def _quote(facts: FactSnapshot) -> str:
        quote = next(iter(facts.quotes.values()), None)
        if quote is None:
            return "正式报价暂时没有生成成功，请确认商品和租期后重试。"
        price = quote.price_snapshot
        return "\n".join(
            (
                "正式报价已经生成：",
                f"- 日租 ¥{price.daily_rate}，计费 {price.billing_days} 天。",
                f"- 租金 ¥{price.rental_amount}，押金 ¥{price.deposit_amount}。",
                f"- 合计 ¥{price.total_amount}。",
                "报价有有效期，确认无误后可以继续预订。",
            )
        )

    @staticmethod
    def _scenario(facts: FactSnapshot) -> str:
        kit = facts.scenario_kits[-1]
        lines = ["我按完整场景需求组合了这套设备："]
        for item in kit.items:
            lines.append(
                f"- {item.product.name} × {item.quantity}："
                f"小计 ¥{item.subtotal_daily_rate}/天。"
            )
        lines.append(
            f"组合日租合计 ¥{kit.total_daily_rate}，预算 ¥{kit.max_daily_budget}，"
            + ("在预算内。" if kit.within_budget else "目前超出预算或缺少必要设备。")
        )
        if kit.missing_roles:
            lines.append("当前目录还缺少部分必要设备，可以调整方案后重新组合。")
        if not kit.availability_checked:
            lines.append("补充完整租期后，我会继续核验整套设备的实时库存。")
        return "\n".join(lines)
