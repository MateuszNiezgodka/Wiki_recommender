import httpx
import pytest

from wiki_recommender.crawler.client import USER_AGENT, ApiRequestError, ApiResponseError, WikiClient


def _client(handler):
    return WikiClient(
        transport=httpx.MockTransport(handler),
        min_interval=0,
        backoff_base=0,
        max_retries=2,
    )


def _sequence(*responses):
    remaining = list(responses)
    seen = []

    def handler(request):
        seen.append(request)
        return remaining.pop(0)

    return handler, seen


def test_identifies_itself_and_yields_to_server_lag():
    handler, seen = _sequence(httpx.Response(200, json={"ok": 1}))
    assert _client(handler).get(action="query") == {"ok": 1}
    assert seen[0].headers["User-Agent"] == USER_AGENT
    assert seen[0].url.params["maxlag"] == "5"


@pytest.mark.parametrize(
    "transient",
    [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(503),
        httpx.Response(
            200, json={"error": {"code": "maxlag", "info": "lagged"}}, headers={"Retry-After": "0"}
        ),
    ],
)
def test_retries_transient_failures(transient):
    handler, seen = _sequence(transient, httpx.Response(200, json={"ok": 1}))
    client = _client(handler)
    assert client.get() == {"ok": 1}
    assert len(seen) == 2
    assert client.retry_count == 1


def test_retries_network_errors():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json={"ok": 1})

    assert _client(handler).get() == {"ok": 1}


def test_gives_up_after_the_retry_budget():
    handler, seen = _sequence(*[httpx.Response(503) for _ in range(3)])
    with pytest.raises(ApiRequestError):
        _client(handler).get()
    assert len(seen) == 3


def test_permanent_api_errors_are_not_retried():
    handler, seen = _sequence(httpx.Response(200, json={"error": {"code": "missingtitle", "info": "x"}}))
    with pytest.raises(ApiResponseError) as caught:
        _client(handler).get()
    assert caught.value.code == "missingtitle"
    assert len(seen) == 1


def test_client_errors_fail_immediately():
    handler, seen = _sequence(httpx.Response(403))
    with pytest.raises(ApiRequestError):
        _client(handler).get()
    assert len(seen) == 1
