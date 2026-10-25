"""Direct Azure OpenAI Chat Completions calls returning JSON, with user-facing error mapping."""
import json
import logging
import time

from openai import APIError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError

from app.results import Usage
from app.settings import Settings


logger = logging.getLogger(__name__)
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh")


class ServiceError(ValueError):
    def __init__(self, message: str, status: int = 502, usage: Usage | None = None):
        super().__init__(message)
        self.status = status
        self.usage = usage or Usage()  # tokens spent by a call that still failed


class OutputLimitError(ServiceError):
    """The model stopped at its output-token limit; callers can retry with smaller input."""


def require_azure(settings: Settings):
    if not settings.configured:
        raise ServiceError("Configure AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT and AZURE_OPENAI_API_KEY, "
                           "then restart the backend.", 503)


def make_client(settings: Settings, timeout: float = 300) -> OpenAI:
    require_azure(settings)
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=timeout, max_retries=2)


def chat_json(settings: Settings, messages: list[dict], *, response_format: dict, operation: str,
              reasoning_effort: str | None = None, max_completion_tokens: int = 32_000,
              client: OpenAI | None = None, timeout: float = 300) -> tuple[dict, Usage, float]:
    """One JSON-mode/structured-output completion. Returns (parsed JSON, usage, seconds)."""
    owned = client is None
    client = client or make_client(settings, timeout)
    started = time.monotonic()
    kwargs = {}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    try:
        response = client.chat.completions.create(
            model=settings.deployment, messages=messages, response_format=response_format,
            max_completion_tokens=max_completion_tokens, **kwargs,
        )
    except AuthenticationError as error:
        raise ServiceError("Azure authentication failed. Check the API key and endpoint.") from error
    except RateLimitError as error:
        raise ServiceError("Azure rate limit reached. Wait a moment and retry.", 429) from error
    except APITimeoutError as error:
        raise ServiceError(f"Azure {operation} timed out.", 504) from error
    except APIError as error:
        detail = str(error).lower()
        logger.warning("Azure %s failed: %s", operation, type(error).__name__)
        if any(term in detail for term in ("context_length", "context length", "maximum context", "too many tokens")):
            raise ServiceError(f"The {operation} input exceeds the model's context window. "
                               "Reduce the chunk size or choose fewer documents.", 413) from error
        if "reasoning_effort" in detail:
            raise ServiceError("This deployment does not support the selected reasoning effort.", 422) from error
        raise ServiceError(f"Azure {operation} request failed ({getattr(error, 'status_code', None) or type(error).__name__}). "
                           "Check the deployment supports Chat Completions with JSON output.") from error
    finally:
        if owned:
            client.close()
    usage = Usage(calls=1, prompt_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
                  completion_tokens=getattr(response.usage, "completion_tokens", 0) or 0)
    elapsed = time.monotonic() - started
    if not response.choices:
        raise ServiceError(f"Azure returned no choices for {operation}.", usage=usage)
    choice = response.choices[0]
    if getattr(choice.message, "refusal", None):
        raise ServiceError(f"The model declined the {operation} request.", usage=usage)
    if choice.finish_reason == "length":
        raise OutputLimitError(f"The model ran out of output tokens during {operation}. "
                               "Use smaller chunks or a narrower schema.", 502, usage=usage)
    if not choice.message.content:
        raise ServiceError(f"Azure returned an empty {operation} response.", usage=usage)
    try:
        return json.loads(choice.message.content), usage, elapsed
    except json.JSONDecodeError as error:
        raise ServiceError(f"The model returned invalid JSON for {operation}.", usage=usage) from error
