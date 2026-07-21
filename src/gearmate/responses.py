from gearmate.actions import AgentAction
from gearmate.tools.contracts import StoreOrderStatus
from gearmate.validation.facts import FactSnapshot


class UserResponseComposer:
    def compose(self, *, action: AgentAction, facts: FactSnapshot) -> str:
        if action.action == "product_search":
            return self._product_search(action, facts)
        if action.action == "product_detail":
            return self._product_detail(facts)
        if action.action == "order_list":
            return self._store_orders(facts)
        if action.action == "order_detail":
            return self._store_order_detail(facts)
        if action.action == "sku_stock":
            return self._sku_stock(facts)
        if action.action == "purchase_prepare":
            return self._purchase_prepare(action, facts)
        return facts.fallback_text()

    @staticmethod
    def _product_search(action: AgentAction, facts: FactSnapshot) -> str:
        products = list(facts.products.values())
        if not products:
            return "暂时没有找到符合条件的商品。可以放宽用途、价格或品牌要求后重试。"
        if action.target_price is not None:
            intro = f"我按目标购买价 ¥{action.target_price} 和使用需求筛选了这些商品："
        elif action.max_price is not None:
            intro = f"我按售价不超过 ¥{action.max_price} 和使用需求筛选了这些商品："
        else:
            intro = "我按商品类型和使用需求筛选了这些候选："

        lines = [intro]
        for index, product in enumerate(products[:4]):
            reasons = "、".join(item.name for item in product.use_cases[:2])
            available_skus = [sku for sku in product.store_skus if sku.enabled]
            detail = (
                f"售价 ¥{min(float(sku.sale_price) for sku in available_skus):.2f} 起"
                if available_skus
                else "暂未配置可售规格"
            )
            if reasons:
                detail += f"，适合{reasons}"
            if available_skus:
                detail += f"，当前库存 {sum(sku.available_quantity for sku in available_skus)} 件"
            prefix = "优先推荐" if index == 0 else "备选"
            lines.append(f"- {prefix} {product.name}：{detail}。")

        lines.append("点开商品卡片可以选择规格和数量，确认后再进入结算，不会由 Agent 自动下单。")
        return "\n".join(lines)

    @staticmethod
    def _product_detail(facts: FactSnapshot) -> str:
        product = next(iter(facts.products.values()), None)
        if product is None:
            return "暂时没有取得这款商品的详细信息，请重新选择商品。"
        reasons = "、".join(item.name for item in product.use_cases[:3])
        skus = [sku for sku in facts.store_skus.values() if sku.product_id == product.product_id]
        lines = [f"这款是 {product.name}（{product.brand} {product.model}）。"]
        if skus:
            lines.append(
                f"目前有 {len(skus)} 种规格，售价 "
                f"¥{min(float(sku.sale_price) for sku in skus):.2f} 起。"
            )
        if reasons:
            lines.append(f"目录中标注的主要适用场景是：{reasons}。")
        lines.append("点开卡片即可选择规格和数量后进入结算。")
        return "\n".join(lines)

    @staticmethod
    def _sku_stock(facts: FactSnapshot) -> str:
        skus = list(facts.store_skus.values())
        if not skus:
            return "这款商品暂时没有可售规格。"
        lines = ["当前可购买规格如下："]
        for sku in skus:
            state = f"库存 {sku.available_quantity} 件" if sku.available_quantity else "暂时缺货"
            lines.append(f"- {sku.sku_name}：售价 ¥{sku.sale_price}，{state}。")
        return "\n".join(lines)

    @classmethod
    def _purchase_prepare(cls, action: AgentAction, facts: FactSnapshot) -> str:
        text = cls._sku_stock(facts)
        if facts.store_skus:
            quantity = action.quantity or 1
            text += f"\n已按 {quantity} 件准备购买选项，请在弹窗中确认规格、数量和收货信息。"
        return text

    @classmethod
    def _store_orders(cls, facts: FactSnapshot) -> str:
        orders = list(facts.store_orders.values())
        if not orders:
            return (
                "当前筛选条件下没有商城订单。"
                if facts.store_order_list_performed
                else "暂时无法取得商城订单。"
            )
        lines = ["这是你最近的商城订单："]
        for order in orders:
            products = "、".join(item.product_name for item in order.items[:2]) or "商城商品"
            quantity = sum(item.quantity for item in order.items)
            lines.append(
                f"- {products}：{cls._store_order_status_label(order.status)}，"
                f"共 {quantity} 件，合计 ¥{order.payable_amount}。"
            )
        return "\n".join(lines)

    @classmethod
    def _store_order_detail(cls, facts: FactSnapshot) -> str:
        order = next(iter(facts.store_orders.values()), None)
        if order is None:
            return "暂时没有取得这笔商城订单。"
        lines = [f"订单状态：{cls._store_order_status_label(order.status)}。"]
        for item in order.items:
            lines.append(
                f"- {item.product_name}：{item.sku_name} × {item.quantity}，"
                f"小计 ¥{item.subtotal}。"
            )
        lines.append(f"应付合计 ¥{order.payable_amount}。")
        if order.carrier and order.tracking_number:
            lines.append(f"物流：{order.carrier}，运单号 {order.tracking_number}。")
        return "\n".join(lines)

    @staticmethod
    def _store_order_status_label(status: StoreOrderStatus) -> str:
        return {
            "PENDING_PAYMENT": "待支付",
            "PAID": "待发货",
            "SHIPPED": "待收货",
            "RECEIVED": "已完成",
            "CANCELLED": "已取消",
            "CLOSED": "已关闭",
        }[status]
