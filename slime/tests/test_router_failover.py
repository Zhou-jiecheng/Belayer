import asyncio
import sys
import types

if "httpx" not in sys.modules:
    httpx_module = types.ModuleType("httpx")

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class _Limits:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class _Timeout:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class _HTTPStatusError(Exception):
        pass

    class _ConnectError(Exception):
        pass

    class _ConnectTimeout(Exception):
        pass

    class _ReadError(Exception):
        pass

    class _ReadTimeout(Exception):
        pass

    class _RemoteProtocolError(Exception):
        pass

    class _WriteError(Exception):
        pass

    class _WriteTimeout(Exception):
        pass

    class _PoolTimeout(Exception):
        pass

    httpx_module.AsyncClient = _AsyncClient
    httpx_module.Limits = _Limits
    httpx_module.Timeout = _Timeout
    httpx_module.HTTPStatusError = _HTTPStatusError
    httpx_module.ConnectError = _ConnectError
    httpx_module.ConnectTimeout = _ConnectTimeout
    httpx_module.ReadError = _ReadError
    httpx_module.ReadTimeout = _ReadTimeout
    httpx_module.RemoteProtocolError = _RemoteProtocolError
    httpx_module.WriteError = _WriteError
    httpx_module.WriteTimeout = _WriteTimeout
    httpx_module.PoolTimeout = _PoolTimeout
    sys.modules["httpx"] = httpx_module

if "uvicorn" not in sys.modules:
    uvicorn_module = types.ModuleType("uvicorn")
    uvicorn_module.run = lambda *args, **kwargs: None
    sys.modules["uvicorn"] = uvicorn_module

if "ray" not in sys.modules:
    ray_module = types.ModuleType("ray")
    sys.modules["ray"] = ray_module

if "slime.utils.misc" not in sys.modules:
    misc_module = types.ModuleType("slime.utils.misc")
    misc_module.load_function = lambda path: path
    sys.modules["slime.utils.misc"] = misc_module

if "fastapi" not in sys.modules:
    fastapi_module = types.ModuleType("fastapi")

    class _FastAPI:
        def post(self, *args, **kwargs):
            del args, kwargs
            return lambda fn: fn

        def get(self, *args, **kwargs):
            del args, kwargs
            return lambda fn: fn

        def api_route(self, *args, **kwargs):
            del args, kwargs
            return lambda fn: fn

        def add_middleware(self, *args, **kwargs):
            del args, kwargs

    class _Request:
        pass

    fastapi_module.FastAPI = _FastAPI
    fastapi_module.Request = _Request
    sys.modules["fastapi"] = fastapi_module

if "fastapi.responses" not in sys.modules:
    fastapi_responses_module = types.ModuleType("fastapi.responses")

    class _JSONResponse:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    fastapi_responses_module.JSONResponse = _JSONResponse
    sys.modules["fastapi.responses"] = fastapi_responses_module

if "starlette.responses" not in sys.modules:
    starlette_responses_module = types.ModuleType("starlette.responses")

    class _Response:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class _StreamingResponse(_Response):
        pass

    starlette_responses_module.Response = _Response
    starlette_responses_module.StreamingResponse = _StreamingResponse
    sys.modules["starlette.responses"] = starlette_responses_module

from slime.router.router import SlimeRouter


def _make_router():
    router = object.__new__(SlimeRouter)
    router.worker_request_counts = {}
    router._worker_selection_cursor = 0
    router.reroute_failed_requests_to_healthy_workers = True
    return router


def test_select_least_loaded_worker_url_round_robins_ties():
    router = _make_router()
    router.worker_request_counts = {
        "http://worker-a": 0,
        "http://worker-b": 0,
        "http://worker-c": 1,
    }

    assert router._select_least_loaded_worker_url() == "http://worker-a"
    assert router._select_least_loaded_worker_url() == "http://worker-b"
    assert router._select_least_loaded_worker_url(exclude_workers={"http://worker-a"}) == "http://worker-b"


def test_resolve_failed_worker_url_prefers_healthy_workers():
    router = _make_router()
    router.worker_request_counts = {
        "http://failed": 2,
        "http://worker-a": 0,
        "http://worker-b": 0,
    }

    async def _unexpected_wait(worker_key: str, failed_url: str):
        raise AssertionError(f"unexpected recovery wait for {worker_key} on {failed_url}")

    router._wait_for_worker_recovery = _unexpected_wait

    assert asyncio.run(router._resolve_failed_worker_url("worker-key", "http://failed")) == "http://worker-a"
    assert asyncio.run(router._resolve_failed_worker_url("worker-key", "http://failed")) == "http://worker-b"


def test_resolve_failed_worker_url_falls_back_when_no_healthy_worker_exists():
    router = _make_router()
    router.worker_request_counts = {"http://failed": 1}
    wait_calls = []

    async def _fake_wait(worker_key: str, failed_url: str):
        wait_calls.append((worker_key, failed_url))
        return "http://recovered"

    router._wait_for_worker_recovery = _fake_wait

    recovered_url = asyncio.run(router._resolve_failed_worker_url("worker-key", "http://failed"))

    assert recovered_url == "http://recovered"
    assert wait_calls == [("worker-key", "http://failed")]
