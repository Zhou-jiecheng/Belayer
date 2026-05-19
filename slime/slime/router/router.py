import argparse
import asyncio
import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response, StreamingResponse

from slime.utils.misc import load_function

logger = logging.getLogger(__name__)


@dataclass
class _GenerateCheckpoint:
    output_ids: list[int] = field(default_factory=list)
    output_token_logprobs: list = field(default_factory=list)
    output_token_ids_logprobs: list = field(default_factory=list)
    output_top_logprobs: list = field(default_factory=list)
    text_prefix: str = ""
    count: int = 0


def run_router(args):
    """
    Run the Slime router with the specified configuration.
    """
    # Initialize the router with tokenizer and lazy worker initialization
    slime_router = SlimeRouter(args, verbose=False)

    # Start the server
    uvicorn.run(slime_router.app, host=args.sglang_router_ip, port=args.sglang_router_port, log_level="info")


class SlimeRouter:
    def __init__(self, args, verbose=False):
        """Initialize the slime-router with SGLang router address"""
        self.args = args
        self.verbose = verbose

        self.app = FastAPI()

        # URL -> Active Request Count (load state)
        self.worker_request_counts: dict[str, int] = {}
        # URL -> stable logical worker key
        self.worker_keys_by_url: dict[str, str] = {}
        # stable logical worker key -> current URL
        self.worker_urls_by_key: dict[str, str] = {}
        # stable logical worker key -> monotonically increasing registration sequence
        self.worker_registration_seq_by_key: dict[str, int] = {}
        # URL -> registration sequence of this concrete worker URL
        self.worker_registration_seq_by_url: dict[str, int] = {}
        # stable logical worker key -> registration event used by /generate recovery waiters
        self.worker_recovery_events_by_key: dict[str, asyncio.Event] = {}
        # URL -> Consecutive Failures
        self.worker_failure_counts: dict[str, int] = {}
        self.max_weight_version = None

        max_connections = getattr(args, "slime_router_max_connections", None)
        if max_connections is None:
            max_connections = (
                args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
            )

        timeout = getattr(args, "slime_router_timeout", None)

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections),
            timeout=httpx.Timeout(timeout),
        )
        self.proxy_max_retries = max(0, int(os.getenv("SLIME_ROUTER_PROXY_MAX_RETRIES", "2")))
        # <=0 means "wait until new worker registration for this worker_key" (no fixed timeout window).
        self.generate_recovery_wait_seconds = float(os.getenv("SLIME_ROUTER_GENERATE_RECOVERY_WAIT_SECONDS", "-1"))
        self.generate_recovery_poll_interval = float(os.getenv("SLIME_ROUTER_GENERATE_RECOVERY_POLL_INTERVAL", "0.5"))
        self.generate_recovery_max_attempts = max(
            0,
            int(os.getenv("SLIME_ROUTER_GENERATE_RECOVERY_MAX_ATTEMPTS", "2")),
        )
        self.disable_token_level_recovery = os.getenv(
            "SLIME_ROUTER_DISABLE_TOKEN_LEVEL_RECOVERY", "0"
        ).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.reroute_failed_requests_to_healthy_workers = os.getenv(
            "SLIME_ROUTER_REROUTE_FAILED_REQUESTS_TO_HEALTHY_WORKERS", "0"
        ).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.generate_chunk_debug_enabled = os.getenv("SLIME_ROUTER_GENERATE_CHUNK_DEBUG", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.generate_idle_debug_seconds = float(os.getenv("SLIME_ROUTER_GENERATE_IDLE_DEBUG_SECONDS", "30"))
        self.generate_chunk_summary_stride = max(1, int(os.getenv("SLIME_ROUTER_GENERATE_CHUNK_SUMMARY_STRIDE", "256")))
        self.generate_recovery_checkpoint_tokens = max(
            1, int(os.getenv("SLIME_ROUTER_GENERATE_CHECKPOINT_TOKENS", "1000"))
        )
        self.generate_coalesce_chunks = max(1, int(os.getenv("SLIME_ROUTER_GENERATE_COALESCE_CHUNKS", "512")))
        self._generate_request_seq = 0
        self._worker_selection_cursor = 0
        self.generate_reroute_poll_interval = float(
            os.getenv("SLIME_ROUTER_GENERATE_REROUTE_POLL_INTERVAL", "0.25")
        )
        logger.info(
            "[slime-router] generate recovery config: max_attempts=%s wait_seconds=%s poll_interval=%s "
            "checkpoint_tokens=%s reroute_poll_interval=%s coalesce_chunks=%s reroute_failed_requests=%s "
            "disable_token_level_recovery=%s",
            self.generate_recovery_max_attempts,
            self.generate_recovery_wait_seconds,
            self.generate_recovery_poll_interval,
            self.generate_recovery_checkpoint_tokens,
            self.generate_reroute_poll_interval,
            self.generate_coalesce_chunks,
            self.reroute_failed_requests_to_healthy_workers,
            self.disable_token_level_recovery,
        )

        self._setup_routes()

        for middleware_path in args.slime_router_middleware_paths or []:
            if self.verbose:
                print(f"[slime-router] Loading middleware from: {middleware_path}")
            middleware = load_function(middleware_path)
            self.app.add_middleware(middleware, router=self)

    @staticmethod
    def _prepare_forward_headers(headers: dict[str, str]) -> dict[str, str]:
        forward_headers = dict(headers)
        for header_name in ("content-length", "host", "transfer-encoding", "connection"):
            forward_headers.pop(header_name, None)
        return forward_headers

    @staticmethod
    def _prepare_response_headers(headers: dict[str, str]) -> dict[str, str]:
        response_headers = dict(headers)
        # The router may buffer / reconstruct the body, so let Starlette
        # compute transport-level headers for the returned response.
        for header_name in ("content-length", "transfer-encoding", "connection"):
            response_headers.pop(header_name, None)
        return response_headers

    def _reset_generate_recovery_state(self) -> tuple[list[int], list, list, list, str, _GenerateCheckpoint]:
        return [], [], [], [], "", _GenerateCheckpoint()

    @staticmethod
    def _capture_generate_checkpoint(
        checkpoint: _GenerateCheckpoint,
        aggregated_output_ids: list[int],
        aggregated_output_token_logprobs: list,
        aggregated_output_token_ids_logprobs: list,
        aggregated_output_top_logprobs: list,
        text_prefix: str,
    ) -> _GenerateCheckpoint:
        return _GenerateCheckpoint(
            output_ids=aggregated_output_ids.copy(),
            output_token_logprobs=aggregated_output_token_logprobs.copy(),
            output_token_ids_logprobs=aggregated_output_token_ids_logprobs.copy(),
            output_top_logprobs=aggregated_output_top_logprobs.copy(),
            text_prefix=text_prefix,
            count=checkpoint.count + 1,
        )

    @staticmethod
    def _restore_generate_checkpoint(
        checkpoint: _GenerateCheckpoint,
    ) -> tuple[list[int], list, list, list, str]:
        return (
            checkpoint.output_ids.copy(),
            checkpoint.output_token_logprobs.copy(),
            checkpoint.output_token_ids_logprobs.copy(),
            checkpoint.output_top_logprobs.copy(),
            checkpoint.text_prefix,
        )

    def _setup_routes(self):
        """Setup all the HTTP routes"""
        # sglang-router api
        self.app.post("/add_worker")(self.add_worker)
        self.app.post("/remove_worker")(self.remove_worker)
        self.app.get("/list_workers")(self.list_workers)
        self.app.post("/retrieve_from_text")(self.retrieve_from_text)
        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)


    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router"""
        if path == "generate":
            return await self._proxy_generate(request, path)
        if path == "generate_nonstream":
            return await self._proxy_generate_nonstream(request, "generate")

        # Get request body and headers
        body = await request.body()
        headers = dict(request.headers)
        query_string = request.url.query
        logger.info(f"Router get path {path}")
        if path == "v1/chat/completions":
            worker_url, worker_seq = self._use_url()
            url = self._build_target_url(worker_url, path, query_string)
            response = None
            try:
                upstream_request = self.client.build_request(
                    request.method,
                    url,
                    content=body,
                    headers=self._prepare_forward_headers(headers),
                )
                response = await self.client.send(upstream_request, stream=True)
                response_headers = self._prepare_response_headers(dict(response.headers))

                async def _stream_upstream_chat_completions():
                    try:
                        async for chunk in response.aiter_raw():
                            yield chunk
                    finally:
                        await response.aclose()
                        self._finish_url(worker_url, worker_seq)

                return StreamingResponse(
                    _stream_upstream_chat_completions(),
                    status_code=response.status_code,
                    headers=response_headers,
                )
            except Exception:
                if response is not None:
                    await response.aclose()
                self._finish_url(worker_url, worker_seq)
                raise

        if path == "v1" or path.startswith("v1/"):
            worker_url, worker_seq = self._use_url()
            url = self._build_target_url(worker_url, path, query_string)
            response = None
            try:
                response = await self.client.request(request.method, url, content=body, headers=headers)
                content = await response.aread()
                content_type = response.headers.get("content-type", "")
                try:
                    data = json.loads(content)
                    return JSONResponse(
                        content=data,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                    )
                except Exception:
                    return Response(
                        content=content,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=content_type or None,
                    )
            finally:
                if response is not None:
                    await response.aclose()
                self._finish_url(worker_url, worker_seq)

        forward_headers = self._prepare_forward_headers(headers)
        excluded_workers = set()

        for attempt in range(self.proxy_max_retries + 1):
            worker_url, worker_seq = self._use_url(exclude_workers=excluded_workers)
            url = self._build_target_url(worker_url, path, query_string)
            response = None
            try:
                upstream_request = self.client.build_request(
                    request.method,
                    url,
                    content=body,
                    headers=forward_headers,
                )
                response = await self.client.send(upstream_request, stream=True)
                raw_chunks = [chunk async for chunk in response.aiter_raw()]
                content = b"".join(raw_chunks)
                return Response(
                    content=content,
                    status_code=response.status_code,
                    headers=self._prepare_response_headers(dict(response.headers)),
                )
            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                httpx.WriteError,
                httpx.WriteTimeout,
                httpx.PoolTimeout,
            ) as e:
                excluded_workers.add(worker_url)
                self.worker_failure_counts[worker_url] = self.worker_failure_counts.get(worker_url, 0) + 1
                if attempt >= self.proxy_max_retries:
                    raise
                logger.warning(
                    "[slime-router] proxy retry %d/%d after %s from worker %s",
                    attempt + 1,
                    self.proxy_max_retries,
                    type(e).__name__,
                    worker_url,
                )
            finally:
                if response is not None:
                    await response.aclose()
                self._finish_url(worker_url, worker_seq)

    async def _proxy_generate(self, request: Request, path: str):
        class _GenerateRerouteSignal(Exception):
            def __init__(self, new_url: str | None, reason: str):
                super().__init__(reason)
                self.new_url = new_url
                self.reason = reason

        body = await request.body()
        headers = dict(request.headers)
        forward_headers = self._prepare_forward_headers(headers)
        payload = json.loads(body) if body else {}
        payload["stream"] = True
        query_string = request.url.query

        worker_url, worker_seq = self._use_url()
        worker_key = self.worker_keys_by_url.get(worker_url, worker_url)
        self._generate_request_seq += 1
        request_id = self._generate_request_seq

        aggregated_output_ids = []
        aggregated_output_token_logprobs = []
        aggregated_output_token_ids_logprobs = []
        aggregated_output_top_logprobs = []
        checkpoint = _GenerateCheckpoint()
        first_prompt_tokens = None
        completed_text_prefix = ""
        current_payload = payload
        current_worker_url = worker_url
        current_worker_seq = worker_seq
        request_start_time = time.monotonic()
        stream_first_chunk_time = None
        stream_last_chunk_time = None
        last_chunk_tokens_total = 0
        chunk_count = 0
        last_idle_log_time = request_start_time
        next_checkpoint_token_target = self.generate_recovery_checkpoint_tokens
        pending_chunk_data = None
        pending_chunk_count = 0
        if self.generate_chunk_debug_enabled:
            logger.info(
                "[slime-router] /generate start request_id=%s worker_key=%s url=%s max_new_tokens=%s",
                request_id,
                worker_key,
                current_worker_url,
                (current_payload.get("sampling_params") or {}).get("max_new_tokens"),
            )

        for recovery_attempt in range(self.generate_recovery_max_attempts + 1):
            response = None
            reserved_worker_url = current_worker_url
            reserved_worker_seq = current_worker_seq
            current_attempt_text = ""
            current_attempt_output_ids = []
            latest_meta_info = None

            def flush_pending_chunk(force: bool = False):
                nonlocal pending_chunk_data, pending_chunk_count
                nonlocal latest_meta_info, first_prompt_tokens, current_attempt_text, last_chunk_tokens_total
                nonlocal aggregated_output_ids, aggregated_output_token_logprobs
                nonlocal aggregated_output_token_ids_logprobs, aggregated_output_top_logprobs
                nonlocal current_attempt_output_ids, checkpoint, next_checkpoint_token_target

                if pending_chunk_data is None:
                    return None

                if not force and pending_chunk_count < self.generate_coalesce_chunks:
                    return None

                data = json.loads(pending_chunk_data)
                pending_chunk_data = None
                pending_chunk_count = 0

                if "error" in data:
                    return JSONResponse(
                        content=data,
                        status_code=response.status_code if response is not None else 500,
                        headers=dict(response.headers) if response is not None else {},
                    )

                meta_info = copy.deepcopy(data.get("meta_info", {}))
                latest_meta_info = meta_info
                if first_prompt_tokens is None and meta_info.get("prompt_tokens") is not None:
                    first_prompt_tokens = meta_info["prompt_tokens"]

                current_attempt_text = data.get("text", current_attempt_text)
                chunk_output_ids, chunk_output_source = self._extract_chunk_output_ids(data, meta_info)
                if chunk_output_ids:
                    last_chunk_tokens_total = len(chunk_output_ids)
                if chunk_output_source != "output_ids":
                    log_fn = logger.info if self.generate_chunk_debug_enabled else logger.debug
                    log_fn(
                        "[slime-router] /generate chunk request_id=%s token source=%s extracted=%s has_output_ids=%s has_output_token_logprobs=%s",
                        request_id,
                        chunk_output_source,
                        len(chunk_output_ids),
                        bool(data.get("output_ids")),
                        bool(meta_info.get("output_token_logprobs")),
                    )
                if not chunk_output_ids:
                    if meta_info.get("output_token_logprobs"):
                        log_fn = logger.info if self.generate_chunk_debug_enabled else logger.debug
                        log_fn(
                            "[slime-router] /generate chunk request_id=%s has output_token_logprobs but extracted no token ids",
                            request_id,
                        )
                    return None

                if (
                    len(chunk_output_ids) >= len(current_attempt_output_ids)
                    and chunk_output_ids[: len(current_attempt_output_ids)] == current_attempt_output_ids
                ):
                    new_output_ids = chunk_output_ids[len(current_attempt_output_ids) :]
                    start_idx = len(current_attempt_output_ids)
                    end_idx = len(chunk_output_ids)
                    current_attempt_output_ids = chunk_output_ids
                else:
                    new_output_ids = chunk_output_ids
                    start_idx = 0
                    end_idx = len(chunk_output_ids)
                    current_attempt_output_ids.extend(chunk_output_ids)

                if not new_output_ids:
                    return None

                aggregated_output_ids.extend(new_output_ids)

                output_token_logprobs = meta_info.get("output_token_logprobs") or []
                if output_token_logprobs:
                    if len(output_token_logprobs) >= end_idx:
                        aggregated_output_token_logprobs.extend(output_token_logprobs[start_idx:end_idx])
                    else:
                        aggregated_output_token_logprobs.extend(output_token_logprobs[-len(new_output_ids) :])

                output_token_ids_logprobs = meta_info.get("output_token_ids_logprobs") or []
                if output_token_ids_logprobs:
                    if len(output_token_ids_logprobs) >= end_idx:
                        aggregated_output_token_ids_logprobs.extend(output_token_ids_logprobs[start_idx:end_idx])
                    else:
                        aggregated_output_token_ids_logprobs.extend(output_token_ids_logprobs[-len(new_output_ids) :])

                output_top_logprobs = meta_info.get("output_top_logprobs") or []
                if output_top_logprobs:
                    if len(output_top_logprobs) >= end_idx:
                        aggregated_output_top_logprobs.extend(output_top_logprobs[start_idx:end_idx])
                    else:
                        aggregated_output_top_logprobs.extend(output_top_logprobs[-len(new_output_ids) :])

                if self.generate_chunk_debug_enabled and len(aggregated_output_ids) % self.generate_chunk_summary_stride == 0:
                    logger.info(
                        "[slime-router] /generate progress request_id=%s worker_key=%s url=%s "
                        "recovery_attempt=%s generated_tokens=%s chunk_count=%s elapsed=%.2fs",
                        request_id,
                        worker_key,
                        reserved_worker_url,
                        recovery_attempt,
                        len(aggregated_output_ids),
                        chunk_count,
                        time.monotonic() - request_start_time,
                    )
                if len(aggregated_output_ids) >= next_checkpoint_token_target:
                    checkpoint = self._capture_generate_checkpoint(
                        checkpoint=checkpoint,
                        aggregated_output_ids=aggregated_output_ids,
                        aggregated_output_token_logprobs=aggregated_output_token_logprobs,
                        aggregated_output_token_ids_logprobs=aggregated_output_token_ids_logprobs,
                        aggregated_output_top_logprobs=aggregated_output_top_logprobs,
                        text_prefix=completed_text_prefix + current_attempt_text,
                    )
                    logger.info(
                        "[slime-router] /generate checkpoint request_id=%s worker_key=%s url=%s "
                        "recovery_attempt=%s checkpoint_index=%s checkpoint_tokens=%s elapsed=%.2fs text_len=%s",
                        request_id,
                        worker_key,
                        reserved_worker_url,
                        recovery_attempt,
                        checkpoint.count,
                        len(checkpoint.output_ids),
                        time.monotonic() - request_start_time,
                        len(checkpoint.text_prefix),
                    )
                    while next_checkpoint_token_target <= len(aggregated_output_ids):
                        next_checkpoint_token_target += self.generate_recovery_checkpoint_tokens

                return None
            try:
                target_url = self._build_target_url(reserved_worker_url, path, query_string)
                logger.info(
                    "[slime-router] proxying /generate request_id=%s to worker_key=%s url=%s recovery_attempt=%s generated_tokens=%s",
                    request_id,
                    worker_key,
                    reserved_worker_url,
                    recovery_attempt,
                    len(aggregated_output_ids),
                )
                async with self.client.stream(
                    request.method,
                    target_url,
                    json=current_payload,
                    headers=forward_headers,
                ) as response:
                    response.raise_for_status()
                    line_iterator = response.aiter_lines()
                    pending_line_task: asyncio.Task | None = None
                    try:
                        while True:
                            current_registered_url = self.worker_urls_by_key.get(worker_key)
                            if current_registered_url and current_registered_url != reserved_worker_url:
                                raise _GenerateRerouteSignal(
                                    new_url=current_registered_url,
                                    reason=(
                                        f"worker switched from {reserved_worker_url} to {current_registered_url} "
                                        f"for worker_key={worker_key}"
                                    ),
                                )
                            if reserved_worker_url not in self.worker_request_counts:
                                raise _GenerateRerouteSignal(
                                    new_url=None,
                                    reason=(
                                        f"worker {reserved_worker_url} removed from router while streaming "
                                        f"for worker_key={worker_key}"
                                    ),
                                )

                            try:
                                if pending_line_task is None:
                                    pending_line_task = asyncio.create_task(line_iterator.__anext__())
                                chunk = await asyncio.wait_for(
                                    asyncio.shield(pending_line_task), timeout=self.generate_reroute_poll_interval
                                )
                                pending_line_task = None
                            except (asyncio.TimeoutError, TimeoutError):
                                flush_result = flush_pending_chunk()
                                if flush_result is not None:
                                    return flush_result
                                now = time.monotonic()
                                last_activity_time = stream_last_chunk_time or request_start_time
                                idle_for = now - last_activity_time
                                if (
                                    self.generate_idle_debug_seconds > 0
                                    and idle_for >= self.generate_idle_debug_seconds
                                    and now - last_idle_log_time >= self.generate_idle_debug_seconds
                                ):
                                    logger.warning(
                                        "[slime-router] /generate idle request_id=%s worker_key=%s url=%s "
                                        "recovery_attempt=%s idle_for=%.2fs elapsed=%.2fs chunk_count=%s "
                                        "generated_tokens=%s last_chunk_tokens_total=%s first_chunk_seen=%s",
                                        request_id,
                                        worker_key,
                                        reserved_worker_url,
                                        recovery_attempt,
                                        idle_for,
                                        now - request_start_time,
                                        chunk_count,
                                        len(aggregated_output_ids),
                                        last_chunk_tokens_total,
                                        stream_first_chunk_time is not None,
                                    )
                                    last_idle_log_time = now
                                continue
                            except StopAsyncIteration:
                                pending_line_task = None
                                flush_result = flush_pending_chunk(force=True)
                                if flush_result is not None:
                                    return flush_result
                                logger.warning(
                                    "[slime-router] /generate stream closed without DONE request_id=%s worker_key=%s "
                                    "url=%s recovery_attempt=%s elapsed=%.2fs chunk_count=%s generated_tokens=%s "
                                    "first_chunk_delay=%.2fs last_chunk_ago=%.2fs",
                                    request_id,
                                    worker_key,
                                    reserved_worker_url,
                                    recovery_attempt,
                                    time.monotonic() - request_start_time,
                                    chunk_count,
                                    len(aggregated_output_ids),
                                    -1.0 if stream_first_chunk_time is None else stream_first_chunk_time - request_start_time,
                                    -1.0 if stream_last_chunk_time is None else time.monotonic() - stream_last_chunk_time,
                                )
                                break

                            if not chunk or not chunk.startswith("data:"):
                                continue
                            now = time.monotonic()
                            if stream_first_chunk_time is None:
                                stream_first_chunk_time = now
                                logger.info(
                                    "[slime-router] /generate first chunk request_id=%s worker_key=%s url=%s "
                                    "recovery_attempt=%s first_chunk_delay=%.2fs",
                                    request_id,
                                    worker_key,
                                    reserved_worker_url,
                                    recovery_attempt,
                                    now - request_start_time,
                                )
                            stream_last_chunk_time = now
                            last_idle_log_time = now
                            chunk_count += 1
                            if chunk == "data: [DONE]":
                                flush_result = flush_pending_chunk(force=True)
                                if flush_result is not None:
                                    return flush_result
                                final_meta_info = latest_meta_info or {}
                                if first_prompt_tokens is not None:
                                    final_meta_info["prompt_tokens"] = first_prompt_tokens
                                final_meta_info["completion_tokens"] = len(aggregated_output_ids)
                                if aggregated_output_token_logprobs:
                                    final_meta_info["output_token_logprobs"] = aggregated_output_token_logprobs
                                if aggregated_output_token_ids_logprobs:
                                    final_meta_info["output_token_ids_logprobs"] = aggregated_output_token_ids_logprobs
                                if aggregated_output_top_logprobs:
                                    final_meta_info["output_top_logprobs"] = aggregated_output_top_logprobs
                                logger.info(
                                    "[slime-router] /generate done request_id=%s worker_key=%s url=%s recovery_attempt=%s "
                                    "generated_tokens=%s elapsed=%.2fs chunk_count=%s first_chunk_delay=%.2fs "
                                    "done_gap=%.2fs checkpoints=%s last_checkpoint_tokens=%s",
                                    request_id,
                                    worker_key,
                                    reserved_worker_url,
                                    recovery_attempt,
                                    len(aggregated_output_ids),
                                    time.monotonic() - request_start_time,
                                    chunk_count,
                                    -1.0 if stream_first_chunk_time is None else stream_first_chunk_time - request_start_time,
                                    -1.0 if stream_last_chunk_time is None else time.monotonic() - stream_last_chunk_time,
                                    checkpoint.count,
                                    len(checkpoint.output_ids),
                                )
                                return JSONResponse(
                                    content={
                                        "text": completed_text_prefix + current_attempt_text,
                                        "output_ids": aggregated_output_ids,
                                        "meta_info": final_meta_info,
                                    },
                                    status_code=response.status_code,
                                    headers=dict(response.headers),
                                )

                            pending_chunk_data = chunk[5:].strip()
                            pending_chunk_count += 1
                            flush_result = flush_pending_chunk()
                            if flush_result is not None:
                                return flush_result
                    finally:
                        if pending_line_task is not None and not pending_line_task.done():
                            pending_line_task.cancel()
                            try:
                                await pending_line_task
                            except asyncio.CancelledError:
                                # Expected during worker handover: the old stream reader is
                                # explicitly cancelled once we reroute or tear down the stale worker.
                                pass
                            except Exception:
                                pass

                raise RuntimeError(f"[slime-router] /generate stream ended unexpectedly for worker_key={worker_key}")
            except _GenerateRerouteSignal as reroute_signal:
                flush_result = flush_pending_chunk(force=True)
                if flush_result is not None:
                    return flush_result
                completed_text_prefix += current_attempt_text
                if recovery_attempt >= self.generate_recovery_max_attempts:
                    logger.error(
                        "[slime-router] /generate reroute exhausted request_id=%s worker_key=%s after %s partial tokens",
                        request_id,
                        worker_key,
                        len(aggregated_output_ids),
                    )
                    raise RuntimeError(
                        f"[slime-router] /generate reroute exhausted for worker_key={worker_key}"
                    ) from reroute_signal

                logger.info(
                    "[slime-router] /generate proactive reroute request_id=%s worker_key=%s url=%s reason=%s",
                    request_id,
                    worker_key,
                    reserved_worker_url,
                    reroute_signal.reason,
                )
                recovered_url = reroute_signal.new_url
                if recovered_url is None:
                    recovered_url = await self._resolve_failed_worker_url(worker_key, reserved_worker_url)
                if recovered_url is None:
                    raise RuntimeError(
                        f"[slime-router] /generate failed to find replacement worker for worker_key={worker_key}"
                    ) from reroute_signal

                if self.disable_token_level_recovery and aggregated_output_ids:
                    logger.info(
                        "[slime-router] /generate reroute restarting from scratch request_id=%s worker_key=%s "
                        "dropping_partial_tokens=%s checkpoints=%s",
                        request_id,
                        worker_key,
                        len(aggregated_output_ids),
                        checkpoint.count,
                    )
                    (
                        aggregated_output_ids,
                        aggregated_output_token_logprobs,
                        aggregated_output_token_ids_logprobs,
                        aggregated_output_top_logprobs,
                        completed_text_prefix,
                        checkpoint,
                    ) = self._reset_generate_recovery_state()
                    next_checkpoint_token_target = self.generate_recovery_checkpoint_tokens
                current_payload = self._build_generate_recovery_payload(payload, aggregated_output_ids)
                logger.info(
                    "[slime-router] /generate reroute recovered request_id=%s worker_key=%s from=%s to=%s generated_tokens=%s "
                    "remaining_max_new_tokens=%s checkpoints=%s",
                    request_id,
                    worker_key,
                    reserved_worker_url,
                    recovered_url,
                    len(aggregated_output_ids),
                    ((current_payload.get("sampling_params") or {}).get("max_new_tokens")),
                    checkpoint.count,
                )
                current_worker_url = recovered_url
                current_worker_seq = self._reserve_url(current_worker_url)
            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                httpx.WriteError,
                httpx.WriteTimeout,
                httpx.PoolTimeout,
                RuntimeError,
            ) as e:
                flush_result = flush_pending_chunk(force=True)
                if flush_result is not None:
                    return flush_result
                self.worker_failure_counts[reserved_worker_url] = self.worker_failure_counts.get(reserved_worker_url, 0) + 1
                completed_text_prefix += current_attempt_text
                if recovery_attempt >= self.generate_recovery_max_attempts:
                    logger.error(
                        "[slime-router] /generate recovery exhausted request_id=%s worker_key=%s after %s partial tokens",
                        request_id,
                        worker_key,
                        len(aggregated_output_ids),
                    )
                    raise
                logger.warning(
                    "[slime-router] /generate stream failed request_id=%s on worker_key=%s url=%s after %s tokens "
                    "with %s; waiting for fast restart (chunk_count=%s first_chunk_delay=%.2fs last_chunk_ago=%.2fs)",
                    request_id,
                    worker_key,
                    reserved_worker_url,
                    len(aggregated_output_ids),
                    type(e).__name__,
                    chunk_count,
                    -1.0 if stream_first_chunk_time is None else stream_first_chunk_time - request_start_time,
                    -1.0 if stream_last_chunk_time is None else time.monotonic() - stream_last_chunk_time,
                )
                if self.disable_token_level_recovery and aggregated_output_ids:
                    logger.info(
                        "[slime-router] /generate recovery restarting from scratch request_id=%s worker_key=%s "
                        "dropping_partial_tokens=%s checkpoints=%s",
                        request_id,
                        worker_key,
                        len(aggregated_output_ids),
                        checkpoint.count,
                    )
                    (
                        aggregated_output_ids,
                        aggregated_output_token_logprobs,
                        aggregated_output_token_ids_logprobs,
                        aggregated_output_top_logprobs,
                        completed_text_prefix,
                        checkpoint,
                    ) = self._reset_generate_recovery_state()
                    next_checkpoint_token_target = self.generate_recovery_checkpoint_tokens
                else:
                    if checkpoint.output_ids and len(checkpoint.output_ids) < len(aggregated_output_ids):
                        logger.info(
                            "[slime-router] /generate recovery rewinding request_id=%s worker_key=%s from_tokens=%s "
                            "to_checkpoint_tokens=%s dropped_tokens=%s checkpoint_index=%s",
                            request_id,
                            worker_key,
                            len(aggregated_output_ids),
                            len(checkpoint.output_ids),
                            len(aggregated_output_ids) - len(checkpoint.output_ids),
                            checkpoint.count,
                        )
                    (
                        aggregated_output_ids,
                        aggregated_output_token_logprobs,
                        aggregated_output_token_ids_logprobs,
                        aggregated_output_top_logprobs,
                        completed_text_prefix,
                    ) = self._restore_generate_checkpoint(checkpoint)
                recovered_url = await self._resolve_failed_worker_url(worker_key, reserved_worker_url)
                if recovered_url is None:
                    raise
                current_payload = self._build_generate_recovery_payload(payload, aggregated_output_ids)
                logger.info(
                    "[slime-router] /generate retry recovered request_id=%s worker_key=%s from=%s to=%s generated_tokens=%s "
                    "remaining_max_new_tokens=%s checkpoints=%s",
                    request_id,
                    worker_key,
                    reserved_worker_url,
                    recovered_url,
                    len(aggregated_output_ids),
                    ((current_payload.get("sampling_params") or {}).get("max_new_tokens")),
                    checkpoint.count,
                )
                current_worker_url = recovered_url
                current_worker_seq = self._reserve_url(current_worker_url)
            finally:
                if response is not None:
                    await response.aclose()
                self._finish_url(reserved_worker_url, reserved_worker_seq)

    async def _proxy_generate_nonstream(self, request: Request, upstream_path: str):
        class _GenerateNonstreamRerouteSignal(Exception):
            def __init__(self, new_url: str | None, reason: str):
                super().__init__(reason)
                self.new_url = new_url
                self.reason = reason

        body = await request.body()
        headers = dict(request.headers)
        forward_headers = self._prepare_forward_headers(headers)
        payload = json.loads(body) if body else {}
        payload["stream"] = False
        query_string = request.url.query
        request_start_time = time.monotonic()

        worker_url, worker_seq = self._use_url()
        worker_key = self.worker_keys_by_url.get(worker_url, worker_url)
        current_worker_url = worker_url
        current_worker_seq = worker_seq

        for recovery_attempt in range(self.generate_recovery_max_attempts + 1):
            response = None
            reserved_worker_url = current_worker_url
            reserved_worker_seq = current_worker_seq
            try:
                current_registered_url = self.worker_urls_by_key.get(worker_key)
                if current_registered_url and current_registered_url != reserved_worker_url:
                    raise _GenerateNonstreamRerouteSignal(
                        new_url=current_registered_url,
                        reason=(
                            f"worker switched from {reserved_worker_url} to {current_registered_url} "
                            f"for worker_key={worker_key}"
                        ),
                    )
                if reserved_worker_url not in self.worker_request_counts:
                    raise _GenerateNonstreamRerouteSignal(
                        new_url=None,
                        reason=f"worker {reserved_worker_url} removed before nonstream request for worker_key={worker_key}",
                    )

                target_url = self._build_target_url(reserved_worker_url, upstream_path, query_string)
                logger.info(
                    "[slime-router] /generate_nonstream start worker_key=%s url=%s max_new_tokens=%s recovery_attempt=%s",
                    worker_key,
                    reserved_worker_url,
                    (payload.get("sampling_params") or {}).get("max_new_tokens"),
                    recovery_attempt,
                )
                response = await self.client.request(
                    request.method,
                    target_url,
                    json=payload,
                    headers=forward_headers,
                )
                if response.status_code >= 500:
                    response.raise_for_status()
                content = await response.aread()
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    data = json.loads(content)
                    logger.info(
                        "[slime-router] /generate_nonstream done worker_key=%s url=%s recovery_attempt=%s status=%s elapsed=%.2fs finish_reason=%s output_tokens=%s",
                        worker_key,
                        reserved_worker_url,
                        recovery_attempt,
                        response.status_code,
                        time.monotonic() - request_start_time,
                        ((data.get("meta_info") or {}).get("finish_reason")),
                        len(data.get("output_ids") or []),
                    )
                    return JSONResponse(
                        content=data,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                    )
                logger.info(
                    "[slime-router] /generate_nonstream done worker_key=%s url=%s recovery_attempt=%s status=%s elapsed=%.2fs non_json_response=%s",
                    worker_key,
                    reserved_worker_url,
                    recovery_attempt,
                    response.status_code,
                    time.monotonic() - request_start_time,
                    content_type or "unknown",
                )
                return Response(
                    content=content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.headers.get("content-type"),
                )
            except _GenerateNonstreamRerouteSignal as reroute_signal:
                logger.warning(
                    "[slime-router] /generate_nonstream reroute request worker_key=%s url=%s reason=%s",
                    worker_key,
                    reserved_worker_url,
                    reroute_signal.reason,
                )
                recovered_url = reroute_signal.new_url
                if recovered_url is None:
                    recovered_url = await self._resolve_failed_worker_url(worker_key, reserved_worker_url)
                if recovered_url is None:
                    raise RuntimeError(
                        f"[slime-router] /generate_nonstream failed to find replacement worker for worker_key={worker_key}"
                    ) from reroute_signal
                logger.info(
                    "[slime-router] /generate_nonstream reroute recovered worker_key=%s from=%s to=%s restarting_request=true",
                    worker_key,
                    reserved_worker_url,
                    recovered_url,
                )
                current_worker_url = recovered_url
                current_worker_seq = self._reserve_url(current_worker_url)
            except (
                httpx.HTTPStatusError,
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                httpx.WriteError,
                httpx.WriteTimeout,
                httpx.PoolTimeout,
                RuntimeError,
            ) as e:
                self.worker_failure_counts[reserved_worker_url] = self.worker_failure_counts.get(reserved_worker_url, 0) + 1
                if recovery_attempt >= self.generate_recovery_max_attempts:
                    logger.error(
                        "[slime-router] /generate_nonstream recovery exhausted worker_key=%s url=%s after %.2fs",
                        worker_key,
                        reserved_worker_url,
                        time.monotonic() - request_start_time,
                    )
                    raise
                logger.warning(
                    "[slime-router] /generate_nonstream request failed worker_key=%s url=%s with %s; waiting for fast restart "
                    "(recovery_attempt=%s elapsed=%.2fs)",
                    worker_key,
                    reserved_worker_url,
                    type(e).__name__,
                    recovery_attempt,
                    time.monotonic() - request_start_time,
                )
                recovered_url = await self._resolve_failed_worker_url(worker_key, reserved_worker_url)
                if recovered_url is None:
                    raise
                logger.info(
                    "[slime-router] /generate_nonstream retry recovered worker_key=%s from=%s to=%s restarting_request=true",
                    worker_key,
                    reserved_worker_url,
                    recovered_url,
                )
                current_worker_url = recovered_url
                current_worker_seq = self._reserve_url(current_worker_url)
            finally:
                if response is not None:
                    await response.aclose()
                self._finish_url(reserved_worker_url, reserved_worker_seq)

    async def add_worker(self, request: Request):
        """Add a new worker to the router.
        Supports providing the URL via query string or JSON body.
        Examples:
        - POST /add_worker?url=http://127.0.0.1:10090
        - POST /add_worker  with body {"url": "http://127.0.0.1:10090"}
        """
        # 1) Prefer query param
        worker_url = request.query_params.get("url") or request.query_params.get("worker_url")
        worker_key = request.query_params.get("worker_key")

        # 2) Fallback to JSON body
        if not worker_url or not worker_key:
            body = await request.body()
            payload = json.loads(body) if body else {}
            worker_url = worker_url or payload.get("url") or payload.get("worker_url")
            worker_key = worker_key or payload.get("worker_key")

        if not worker_url:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )
        worker_key = worker_key or worker_url

        # Add if new, keep a simple request count per worker
        before_count = len(self.worker_request_counts)
        if worker_url not in self.worker_request_counts:
            self.worker_request_counts[worker_url] = 0
            self.worker_failure_counts[worker_url] = 0
            if self.verbose:
                print(f"[slime-router] Added new worker: {worker_url}")
        else:
            self.worker_failure_counts[worker_url] = 0

        self.worker_keys_by_url[worker_url] = worker_key
        self.worker_urls_by_key[worker_key] = worker_url
        seq = self.worker_registration_seq_by_key.get(worker_key, 0) + 1
        self.worker_registration_seq_by_key[worker_key] = seq
        self.worker_registration_seq_by_url[worker_url] = seq
        recovery_event = self.worker_recovery_events_by_key.get(worker_key)
        if recovery_event is None:
            recovery_event = asyncio.Event()
            self.worker_recovery_events_by_key[worker_key] = recovery_event
        recovery_event.set()
        logger.info(
            "[slime-router] add_worker url=%s worker_key=%s seq=%s before=%s after=%s",
            worker_url,
            worker_key,
            seq,
            before_count,
            len(self.worker_request_counts),
        )

        return {"status": "success", "worker_urls": self.worker_request_counts}

    async def remove_worker(self, request: Request):
        """Remove a worker from the router.

        Supports providing the URL via query string or JSON body.
        Examples:
        - POST /remove_worker?url=http://127.0.0.1:10090
        - POST /remove_worker with body {"url": "http://127.0.0.1:10090"}
        """
        worker_url = request.query_params.get("url") or request.query_params.get("worker_url")
        worker_key = request.query_params.get("worker_key")

        if not worker_url:
            body = await request.body()
            payload = json.loads(body) if body else {}
            worker_url = worker_url or payload.get("url") or payload.get("worker_url")
            worker_key = worker_key or payload.get("worker_key")

        if not worker_url:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )
        worker_key = worker_key or self.worker_keys_by_url.get(worker_url) or worker_url

        before_count = len(self.worker_request_counts)
        existed = worker_url in self.worker_request_counts
        self.worker_request_counts.pop(worker_url, None)
        self.worker_failure_counts.pop(worker_url, None)
        self.worker_registration_seq_by_url.pop(worker_url, None)
        self.worker_keys_by_url.pop(worker_url, None)
        if self.worker_urls_by_key.get(worker_key) == worker_url:
            self.worker_urls_by_key.pop(worker_key, None)

        if self.verbose and existed:
            print(f"[slime-router] Removed worker: {worker_url}")
        logger.info(
            "[slime-router] remove_worker url=%s worker_key=%s existed=%s before=%s after=%s",
            worker_url,
            worker_key,
            existed,
            before_count,
            len(self.worker_request_counts),
        )

        return {
            "status": "success",
            "removed": existed,
            "worker_urls": list(self.worker_request_counts.keys()),
        }

    async def list_workers(self, request: Request):
        """List registered workers.

        Returns all registered workers.
        """
        return {"urls": list(self.worker_request_counts.keys())}

    async def retrieve_from_text(self, request: Request):
        """Get token information from text input"""
        body = await request.body()
        payload = json.loads(body) if body else {}

        text = payload.get("text", "")

        # Use radix tree's retrieve_from_text method (no need to fetch weight version here)
        token_ids, logp, loss_mask = self.radix_tree.retrieve_from_text(text, return_logprob=True)

        # Handle the result based on whether logp was requested
        result = {
            "tokens": token_ids,  # token IDs
            "response": text,  # The input text
            "loss_mask": loss_mask,  # Loss mask for the tokens
            "token_length": len(token_ids),
            "loss_mask_length": len(loss_mask),
            "rollout_logp": logp,
        }

        return result

    def _use_url(self, exclude_workers: set[str] | None = None) -> tuple[str, int]:
        """Select worker URL with minimal active requests."""

        exclude_workers = exclude_workers or set()
        url = self._select_least_loaded_worker_url(exclude_workers=exclude_workers)

        self.worker_request_counts[url] += 1
        seq = self.worker_registration_seq_by_url.get(url, 0)
        logger.debug(
            "[slime-router] reserve via use_url url=%s seq=%s count=%s excluded=%s",
            url,
            seq,
            self.worker_request_counts[url],
            len(exclude_workers),
        )
        return url, seq

    def _select_least_loaded_worker_url(self, exclude_workers: set[str] | None = None) -> str:
        exclude_workers = exclude_workers or set()
        candidates = [worker for worker in self.worker_request_counts if worker not in exclude_workers]
        if not candidates:
            raise RuntimeError("No workers available in the pool")

        min_request_count = min(self.worker_request_counts[worker] for worker in candidates)
        least_loaded_workers = sorted(
            worker for worker in candidates if self.worker_request_counts[worker] == min_request_count
        )
        selected_index = self._worker_selection_cursor % len(least_loaded_workers)
        self._worker_selection_cursor += 1
        return least_loaded_workers[selected_index]

    async def _resolve_failed_worker_url(self, worker_key: str, failed_url: str) -> str | None:
        if self.reroute_failed_requests_to_healthy_workers:
            try:
                healthy_url = self._select_least_loaded_worker_url(exclude_workers={failed_url})
            except RuntimeError:
                healthy_url = None

            if healthy_url is not None:
                logger.info(
                    "[slime-router] rerouting failed request for worker_key=%s away from failed_url=%s to healthy_url=%s",
                    worker_key,
                    failed_url,
                    healthy_url,
                )
                return healthy_url

            logger.info(
                "[slime-router] no alternate healthy worker available for worker_key=%s failed_url=%s; "
                "falling back to worker recovery wait",
                worker_key,
                failed_url,
            )

        return await self._wait_for_worker_recovery(worker_key, failed_url)

    def _reserve_url(self, url: str) -> int:
        if url not in self.worker_request_counts:
            raise RuntimeError(f"Cannot reserve unknown worker url {url}")
        self.worker_request_counts[url] += 1
        seq = self.worker_registration_seq_by_url.get(url, 0)
        logger.debug(
            "[slime-router] reserve explicit url=%s seq=%s count=%s",
            url,
            seq,
            self.worker_request_counts[url],
        )
        return seq

    def _finish_url(self, url: str, expected_seq: int | None = None):
        """Mark the request to the given URL as finished"""
        if url not in self.worker_request_counts:
            logger.info("[slime-router] finish_url ignored for stale worker url=%s", url)
            return
        current_seq = self.worker_registration_seq_by_url.get(url, 0)
        if expected_seq is not None and current_seq != expected_seq:
            logger.info(
                "[slime-router] finish_url ignored for stale reservation url=%s expected_seq=%s current_seq=%s count=%s",
                url,
                expected_seq,
                current_seq,
                self.worker_request_counts[url],
            )
            return
        self.worker_request_counts[url] -= 1
        logger.debug(
            "[slime-router] finish url=%s expected_seq=%s current_seq=%s count=%s",
            url,
            expected_seq,
            current_seq,
            self.worker_request_counts[url],
        )
        assert self.worker_request_counts[url] >= 0, f"URL {url} count went negative"

    def _build_target_url(self, worker_url: str, path: str, query_string: str) -> str:
        url = f"{worker_url}/{path}"
        if query_string:
            url = f"{url}?{query_string}"
        return url

    @staticmethod
    def _to_int_token_id(value) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("-"):
                body = stripped[1:]
            else:
                body = stripped
            if body.isdigit():
                return int(stripped)
        return None

    def _normalize_output_ids(self, maybe_ids) -> list[int]:
        if not isinstance(maybe_ids, list):
            return []
        normalized = []
        for token_id in maybe_ids:
            parsed = self._to_int_token_id(token_id)
            if parsed is not None:
                normalized.append(parsed)
        return normalized

    def _extract_output_ids_from_meta_info(self, meta_info: dict) -> list[int]:
        output_token_logprobs = meta_info.get("output_token_logprobs") or []
        if not isinstance(output_token_logprobs, list):
            return []

        extracted_ids = []
        for entry in output_token_logprobs:
            token_id = None
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                token_id = self._to_int_token_id(entry[1])
            elif isinstance(entry, dict):
                token_id = self._to_int_token_id(
                    entry.get("token_id")
                    if "token_id" in entry
                    else entry.get("id")
                )

            if token_id is not None:
                extracted_ids.append(token_id)

        return extracted_ids

    def _extract_chunk_output_ids(self, data: dict, meta_info: dict) -> tuple[list[int], str]:
        chunk_output_ids = self._normalize_output_ids(data.get("output_ids"))
        if chunk_output_ids:
            return chunk_output_ids, "output_ids"

        extracted_ids = self._extract_output_ids_from_meta_info(meta_info)
        if extracted_ids:
            return extracted_ids, "meta_info.output_token_logprobs"
        return [], "none"

    async def _wait_for_worker_recovery(self, worker_key: str, failed_url: str) -> str | None:
        failed_seq = self.worker_registration_seq_by_url.get(failed_url, 0)
        observed_seq = self.worker_registration_seq_by_key.get(worker_key, 0)
        effective_failed_seq = max(failed_seq, observed_seq)
        logger.info(
            "[slime-router] waiting worker recovery worker_key=%s failed_url=%s failed_seq=%s observed_seq=%s wait_seconds=%s",
            worker_key,
            failed_url,
            failed_seq,
            observed_seq,
            self.generate_recovery_wait_seconds,
        )
        recovery_event = self.worker_recovery_events_by_key.get(worker_key)
        if recovery_event is None:
            recovery_event = asyncio.Event()
            self.worker_recovery_events_by_key[worker_key] = recovery_event

        deadline = None
        if self.generate_recovery_wait_seconds > 0:
            deadline = asyncio.get_event_loop().time() + self.generate_recovery_wait_seconds

        while True:
            current_url = self.worker_urls_by_key.get(worker_key)
            current_seq = self.worker_registration_seq_by_key.get(worker_key, 0)
            if current_url and current_seq > effective_failed_seq:
                logger.info(
                    "[slime-router] worker_key=%s recovered by registration event on url=%s (seq=%s, failed_url=%s, failed_seq=%s, observed_seq=%s)",
                    worker_key,
                    current_url,
                    current_seq,
                    failed_url,
                    failed_seq,
                    observed_seq,
                )
                return current_url

            if current_seq > observed_seq:
                observed_seq = current_seq

            if deadline is None:
                await recovery_event.wait()
                recovery_event.clear()
                continue

            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.warning(
                    "[slime-router] timed out waiting %.1fs for worker_key=%s to recover after failure on %s",
                    self.generate_recovery_wait_seconds,
                    worker_key,
                    failed_url,
                )
                return None
            try:
                await asyncio.wait_for(recovery_event.wait(), timeout=min(self.generate_recovery_poll_interval, remaining))
                recovery_event.clear()
            except (asyncio.TimeoutError, TimeoutError):
                continue

    def _build_generate_recovery_payload(self, original_payload: dict, generated_output_ids: list[int]) -> dict:
        payload = copy.deepcopy(original_payload)
        payload["stream"] = True
        input_ids = list(payload.get("input_ids") or [])
        if self.disable_token_level_recovery:
            payload["input_ids"] = input_ids
        else:
            payload["input_ids"] = input_ids + generated_output_ids
        sampling_params = copy.deepcopy(payload.get("sampling_params") or {})
        max_new_tokens = sampling_params.get("max_new_tokens")
        if max_new_tokens is not None and not self.disable_token_level_recovery:
            sampling_params["max_new_tokens"] = max(0, int(max_new_tokens) - len(generated_output_ids))
        payload["sampling_params"] = sampling_params
        return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--sglang-host", type=str, required=True)
    parser.add_argument("--sglang-port", type=int, required=True)
    parser.add_argument("--tokenizer-name", type=str, help="Name of the tokenizer to use for tokenization")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Run the router
    run_router(args)
