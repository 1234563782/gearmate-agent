from gearmate.prompts.loader import load_system_prompt


def test_system_prompt_keeps_internal_ids_out_of_user_responses() -> None:
    prompt = load_system_prompt()

    assert prompt.version == "2.0.1"
    assert "不得向用户展示" in prompt.content
    assert "不得展示商品 ID、SKU ID、订单 ID 等内部标识" in prompt.content
    assert "Agent 不得自动调用下单或支付接口" in prompt.content
    assert "必须同时给出工具返回的商品 ID" not in prompt.content
