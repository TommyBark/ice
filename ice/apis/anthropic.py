from collections.abc import Mapping
from typing import Optional
from typing import Union
from typing import List

import httpx
from httpx import Response
from httpx import TimeoutException
from structlog.stdlib import get_logger
from tenacity import retry
from tenacity.retry import retry_any
from tenacity.retry import retry_if_exception
from tenacity.retry import retry_if_exception_type
from tenacity.wait import wait_random_exponential

from ice.cache import diskcache
from ice.settings import settings
from ice.trace import add_fields
from ice.trace import trace

log = get_logger()


class RateLimitError(Exception):
    def __init__(self, response: httpx.Response):
        self.response = response
        try:
            message = response.json()["error"]["message"]
        except Exception:
            message = response.text[:100]
        super().__init__(message)


def log_attempt_number(retry_state):
    if retry_state.attempt_number > 1:
        exception = retry_state.outcome.exception()
        exception_name = exception.__class__.__name__
        exception_message = str(exception)
        log.warning(f"Retrying ({exception_name}: {exception_message}): ")


def make_headers() -> dict[str, str]:
    headers = {
        "content-type": "application/json",
        "x-api-key": f"{settings.ANTHROPIC_API_KEY}",
        "anthropic-version": "2023-06-01",
    }
    return headers


RETRYABLE_STATUS_CODES = {408, 429, 502, 503, 504}
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"


def is_retryable_HttpError(e: BaseException) -> bool:
    return (
        isinstance(e, httpx.HTTPStatusError)
        and e.response.status_code in RETRYABLE_STATUS_CODES
    )


class TooLongRequestError(ValueError):
    def __init__(self, prompt: str = "", detail: str = ""):
        self.prompt = prompt
        self.detail = detail
        super().__init__(self.detail)


def raise_if_too_long_error(prompt: object, response: Response) -> None:
    # Raise something more specific than
    # a generic status error if we have exceeded
    # an OpenAI model's context window
    if not isinstance(prompt, str) or response.status_code != 400:
        return None
    try:
        body = response.json()
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    message = body.get("error", dict).get("message", "")
    if not isinstance(message, str):
        return None
    # This is a bit fragile, but since OpenAI can
    # return 400s for other reasons, checking
    # the message seems like the only real
    # way to tell.
    if "maximum context length" not in message:
        return None
    raise TooLongRequestError(prompt=prompt, detail=message)


@diskcache()
@retry(
    retry=retry_any(
        retry_if_exception(is_retryable_HttpError),
        retry_if_exception_type(TimeoutException),
        retry_if_exception_type(RateLimitError),
    ),
    wait=wait_random_exponential(min=1),
    after=log_attempt_number,
)
async def _post(
    endpoint: str, json: dict, timeout: Optional[float] = None, cache_id: int = 0
) -> Union[dict, TooLongRequestError]:
    """Send a POST request to the Anthropic API and return the JSON response."""
    cache_id  # unused
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    _timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=timeout)
    async with httpx.AsyncClient(limits=limits) as client:
        headers = make_headers()
        response = await client.post(
            f"{ANTHROPIC_BASE_URL}/{endpoint}",
            json=json,
            headers=headers,
            timeout=_timeout or 60,
        )
        if response.status_code == 429:
            raise RateLimitError(response)
        response.raise_for_status()
        return response.json()


def extract_total_tokens(response: dict) -> int:
    return response.get("usage", {}).get("total_tokens", 0)


@trace
async def anthropic_complete(
    prompt: str,
    stop: Optional[str] = "\n",
    top_p: float = 1,
    temperature: float = 0,
    model: str = "claude-instant-1.2",
    max_tokens: int = 256,
    logprobs: Optional[int] = None,
    logit_bias: Optional[Mapping[str, Union[int, float]]] = None,
    n: Optional[int] = None,
    echo: Optional[bool] = None,
    cache_id: int = 0,  # for repeated non-deterministic sampling using caching
) -> dict:
    """Send a completion request to the Anthropic API and return the JSON response."""
    # ANTHROPIC_API doesn't support logprobs, logit_bias, n, or echo

    # Completion prompt has to contain roles Human and Assistant
    prompt = "\n\nHuman:" + prompt
    if "Assistant:" not in prompt:
        prompt += "\n\nAssistant:"

    params = {
        "prompt": prompt,
        "top_p": top_p,
        "temperature": temperature,
        "model": model,
        "max_tokens_to_sample": max_tokens,
    }
    if stop is not None:
        stop_sequences = list(stop)
        params["stop_sequences"] = stop_sequences
    response = await _post(
        "complete",
        params,
        cache_id=cache_id,
    )
    if isinstance(response, TooLongRequestError):
        raise response

    add_fields(davinci_equivalent_tokens=extract_total_tokens(response))
    return response


@trace
async def anthropic_chatcomplete(
    messages: list[dict[str, str]],
    stop: Optional[str] = "\n",
    top_p: float = 1,
    temperature: float = 0,
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 256,
    logit_bias: Optional[Mapping[str, Union[int, float]]] = None,
    n: int = 1,
    cache_id: int = 0,  # for repeated non-deterministic sampling using caching
) -> dict:
    """Send a completion request to the Anthropic API and return the JSON response."""
    pass


@trace
async def anthropic_embedding(
    input: Union[str, list[str]],
    model: str,
    cache_id: int = 0,  # for repeated non-deterministic sampling using caching
) -> dict:
    """Send an embedding request to the Anthropic API and return the JSON response."""
    pass
