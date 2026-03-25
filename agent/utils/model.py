import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

OPENAI_RESPONSES_WS_BASE_URL = "wss://api.openai.com/v1"

KONG_API_URL = os.environ.get("KONG_API_URL", "")
KONG_API_KEY = os.environ.get("KONG_API_KEY", "")
KONG_MODEL_ROUTE = os.environ.get("KONG_MODEL_ROUTE", "sonnet-4.6/v2.0.0")


class _KongChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that remaps 'system' role to 'developer' for Kong."""

    def _get_request_payload(
        self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        for message in payload.get("messages", []):
            if message.get("role") == "system":
                message["role"] = "developer"
            content = message.get("content")
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif isinstance(block, str):
                        text_parts.append(block)
                message["content"] = "\n".join(text_parts)
        return payload


def make_model(model_id: str, **kwargs: dict):
    if KONG_API_URL and KONG_API_KEY:
        base_url = KONG_API_URL.rstrip("/") + "/" + KONG_MODEL_ROUTE.strip("/")
        filtered = {k: v for k, v in kwargs.items() if k in ("temperature", "max_tokens")}
        return _KongChatOpenAI(
            base_url=base_url,
            api_key=KONG_API_KEY,
            default_headers={"api-key": KONG_API_KEY},
            model="kong-routed",
            **filtered,
        )

    model_kwargs = kwargs.copy()

    if model_id.startswith("openai:"):
        model_kwargs["base_url"] = OPENAI_RESPONSES_WS_BASE_URL
        model_kwargs["use_responses_api"] = True

    return init_chat_model(model=model_id, **model_kwargs)
