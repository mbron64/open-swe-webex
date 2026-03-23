import os

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

OPENAI_RESPONSES_WS_BASE_URL = "wss://api.openai.com/v1"

KONG_API_URL = os.environ.get("KONG_API_URL", "")
KONG_API_KEY = os.environ.get("KONG_API_KEY", "")
KONG_MODEL_ROUTE = os.environ.get("KONG_MODEL_ROUTE", "sonnet-4.6/v2.0.0")


def make_model(model_id: str, **kwargs: dict):
    if KONG_API_URL and KONG_API_KEY:
        base_url = KONG_API_URL.rstrip("/") + "/" + KONG_MODEL_ROUTE.strip("/")
        filtered = {k: v for k, v in kwargs.items() if k in ("temperature", "max_tokens")}
        return ChatOpenAI(
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
