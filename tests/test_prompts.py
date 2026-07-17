from gearmate.prompts.loader import load_system_prompt


def test_system_prompt_keeps_internal_ids_out_of_user_responses() -> None:
    prompt = load_system_prompt()

    assert prompt.version == "1.1.1"
    assert "不得向用户展示" in prompt.content
    assert "不得展示商品 ID、报价 ID、订单 ID 等内部标识" in prompt.content
    assert "必须同时给出工具返回的商品 ID" not in prompt.content
