from gearmate.config import Settings
from gearmate.llm.openai_compatible import (
    OpenAICompatibleChatModel,
    OpenAICompatibleConfig,
)
from gearmate.llm.port import ChatModelPort


def build_chat_model(settings: Settings) -> ChatModelPort:
    if settings.model_provider == "openai-compatible":
        return OpenAICompatibleChatModel(OpenAICompatibleConfig.from_settings(settings))
    raise ValueError(f"Unsupported model provider: {settings.model_provider}")
