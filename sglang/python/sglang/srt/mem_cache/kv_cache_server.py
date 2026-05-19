import argparse
import json
import logging
import os
import socket
import struct
from multiprocessing.reduction import ForkingPickler
from typing import Dict, List, Optional

import torch
from transformers import AutoConfig

from sglang.srt.utils import get_available_gpu_memory
from sglang.srt.utils.hf_transformers_utils import get_hf_text_config

logger = logging.getLogger(__name__)


def _load_decoder_config_from_json(model_path: str) -> dict:
    """Load text-decoder related fields directly from config.json when available."""
    if not model_path:
        return {}
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
    except Exception as e:
        logger.warning("Failed to read %s: %s", config_path, e)
        return {}

    candidates = [raw_config]
    for key in ("text_config", "language_config", "llm_config", "model_config"):
        nested = raw_config.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)

    fields = {}
    for candidate in candidates:
        for key in (
            "num_key_value_heads",
            "num_attention_heads",
            "hidden_size",
            "head_dim",
            "num_hidden_layers",
            "max_position_embeddings",
        ):
            if key not in fields and key in candidate:
                fields[key] = candidate[key]
    return fields


def send_obj(conn, obj):
    """Send object with length prefix."""
    buf = ForkingPickler.dumps(obj)
    conn.sendall(struct.pack("!I", len(buf)))
    conn.sendall(buf)


def recv_obj(conn):
    """Receive object with length prefix."""
    size_data = conn.recv(4)
    if not size_data:
        return None
    size = struct.unpack("!I", size_data)[0]

    data = b""
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            break
        data += packet

    return ForkingPickler.loads(data)


class KVCacheServer:
    def __init__(
        self,
        socket_path: str,
        gpu_ids: List[int],
        max_total_num_tokens: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        max_num_reqs: int,
        max_context_len: int,
    ):
        self.socket_path = socket_path
        self.gpu_ids = gpu_ids
        self.tp_size = len(gpu_ids)
        self.max_total_num_tokens = max_total_num_tokens
        self.page_size = page_size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.max_num_reqs = max_num_reqs
        self.max_context_len = max_context_len

        # Per-TP-rank shared handles.
        self.data_per_tp_rank = {}

    def allocate(self):
        logger.info(
            "Allocating KV cache for TP group %s (tp_size=%s)",
            self.gpu_ids,
            self.tp_size,
        )
        logger.info(
            "max_total_num_tokens=%s page_size=%s dtype=%s head_num=%s head_dim=%s layer_num=%s max_num_reqs=%s max_context_len=%s",
            self.max_total_num_tokens,
            self.page_size,
            self.dtype,
            self.head_num,
            self.head_dim,
            self.layer_num,
            self.max_num_reqs,
            self.max_context_len,
        )
        from torch.multiprocessing.reductions import reduce_tensor

        for tp_rank, gpu_id in enumerate(self.gpu_ids):
            device = f"cuda:{gpu_id}"
            logger.info("Allocating TP rank %s on %s", tp_rank, device)
            with torch.cuda.device(device):
                head_num_per_tp = self.head_num // self.tp_size
                if head_num_per_tp == 0:
                    head_num_per_tp = 1
                    logger.warning(
                        "head_num (%s) < tp_size (%s), forcing head_num_per_tp=1",
                        self.head_num,
                        self.tp_size,
                    )

                req_to_token = torch.zeros(
                    (self.max_num_reqs, self.max_context_len),
                    dtype=torch.int32,
                    device=device,
                )

                buffer_size = self.max_total_num_tokens + self.page_size
                k_buffer = [
                    torch.zeros(
                        (buffer_size, head_num_per_tp, self.head_dim),
                        dtype=self.dtype,
                        device=device,
                    )
                    for _ in range(self.layer_num)
                ]
                v_buffer = [
                    torch.zeros(
                        (buffer_size, head_num_per_tp, self.head_dim),
                        dtype=self.dtype,
                        device=device,
                    )
                    for _ in range(self.layer_num)
                ]
                torch.cuda.synchronize()

                self.data_per_tp_rank[tp_rank] = {
                    "req_to_token": reduce_tensor(req_to_token),
                    "k_buffer": [reduce_tensor(t) for t in k_buffer],
                    "v_buffer": [reduce_tensor(t) for t in v_buffer],
                    "max_total_num_tokens": self.max_total_num_tokens,
                    "max_num_reqs": self.max_num_reqs,
                    "tp_rank": tp_rank,
                    "tp_size": self.tp_size,
                    "gpu_id": gpu_id,
                    "head_num_per_tp": head_num_per_tp,
                }
            logger.info(
                "TP rank %s prepared (%s layers, head_num_per_tp=%s)",
                tp_rank,
                len(k_buffer),
                head_num_per_tp,
            )

    def run(self):
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.socket_path)
        server.listen(5)
        logger.info("KVCacheServer listening on %s", self.socket_path)

        while True:
            try:
                conn, _ = server.accept()
                request = recv_obj(conn)
                if request is None:
                    conn.close()
                    continue

                tp_rank = request.get("tp_rank", 0)
                if tp_rank not in self.data_per_tp_rank:
                    send_obj(conn, {"error": f"Invalid tp_rank {tp_rank}"})
                    conn.close()
                    continue

                send_obj(conn, self.data_per_tp_rank[tp_rank])
                conn.close()
            except Exception as e:
                logger.exception("Error handling KV cache request: %s", e)


def get_kv_cache_from_server(
    socket_path: str, tp_rank: int = 0
) -> Optional[Dict[str, torch.Tensor]]:
    """Client helper to retrieve KV cache handles for one TP rank."""
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(socket_path)
        send_obj(client, {"tp_rank": tp_rank})
        data = recv_obj(client)
        client.close()

        if data is None:
            logger.error("Received None from KV cache server")
            return None
        if "error" in data:
            logger.error("KV cache server error: %s", data["error"])
            return None

        def rebuild(handle):
            func, args = handle
            return func(*args)

        req_to_token = rebuild(data["req_to_token"])
        k_buffer = [rebuild(h) for h in data["k_buffer"]]
        v_buffer = [rebuild(h) for h in data["v_buffer"]]
        logger.info(
            "Retrieved KV cache for tp_rank=%s with %s layers",
            tp_rank,
            len(k_buffer),
        )
        return {
            "req_to_token": req_to_token,
            "k_buffer": k_buffer,
            "v_buffer": v_buffer,
            "max_total_num_tokens": data["max_total_num_tokens"],
            "max_num_reqs": data["max_num_reqs"],
            "tp_rank": data.get("tp_rank", 0),
            "tp_size": data.get("tp_size", 1),
            "head_num_per_tp": data.get("head_num_per_tp"),
        }
    except Exception as e:
        logger.exception("Failed to get KV cache from server: %s", e)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="KV Cache Server")
    parser.add_argument("--socket-path", type=str, required=True)
    parser.add_argument(
        "--gpu-id",
        "--gpu-ids",
        dest="gpu_ids",
        type=str,
        required=True,
        help="Comma-separated GPU IDs for TP group, e.g. '0,1,2,3'",
    )
    parser.add_argument("--max-total-num-tokens", type=int)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--head-num", type=int, help="Total head_num before TP sharding")
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--layer-num", type=int)
    parser.add_argument("--max-num-reqs", type=int, required=False)
    parser.add_argument("--max-context-len", type=int, required=False)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--mem-fraction-static", type=float, default=0.9)
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    tp_size = len(gpu_ids)
    logger.info("TP group: %s (tp_size=%s)", gpu_ids, tp_size)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "fp8_e5m2": torch.float8_e5m2,
        "fp8_e4m3": torch.float8_e4m3fn,
    }

    if args.model_path:
        json_config = _load_decoder_config_from_json(args.model_path)
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        text_config = get_hf_text_config(config)

        head_num = getattr(text_config, "num_key_value_heads", None)
        if head_num is None:
            head_num = getattr(text_config, "num_attention_heads", None)
        if head_num is None:
            head_num = getattr(config, "num_key_value_heads", None)
        if head_num is None:
            head_num = getattr(config, "num_attention_heads", None)
        if head_num is None:
            head_num = json_config.get("num_key_value_heads") or json_config.get(
                "num_attention_heads"
            )
        if head_num is None:
            raise AttributeError(
                f"Cannot infer num_key_value_heads/num_attention_heads from config type={type(config).__name__}"
            )

        hidden_size = getattr(text_config, "hidden_size", None)
        num_attention_heads = getattr(text_config, "num_attention_heads", None)
        if hidden_size is None:
            hidden_size = getattr(config, "hidden_size", None)
        if num_attention_heads is None:
            num_attention_heads = getattr(config, "num_attention_heads", None)
        if hidden_size is None:
            hidden_size = json_config.get("hidden_size")
        if num_attention_heads is None:
            num_attention_heads = json_config.get("num_attention_heads")
        if hidden_size is None or num_attention_heads is None:
            raise AttributeError(
                f"Cannot infer hidden_size/num_attention_heads from config type={type(config).__name__}"
            )

        head_dim = getattr(text_config, "head_dim", None)
        if head_dim is None:
            head_dim = json_config.get("head_dim")
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads

        layer_num = getattr(text_config, "num_hidden_layers", None)
        if layer_num is None:
            layer_num = getattr(config, "num_hidden_layers", None)
        if layer_num is None:
            layer_num = json_config.get("num_hidden_layers")
        if layer_num is None:
            raise AttributeError(
                f"Cannot infer num_hidden_layers from config type={type(config).__name__}"
            )

        max_context_len = getattr(text_config, "max_position_embeddings", None)
        if max_context_len is None:
            max_context_len = getattr(config, "max_position_embeddings", None)
        if max_context_len is None:
            max_context_len = json_config.get("max_position_embeddings")
        if max_context_len is None:
            max_context_len = 32768

        available_gpu_memory = get_available_gpu_memory("cuda", gpu_ids[0])
        dtype_size = 2
        if args.dtype == "float32":
            dtype_size = 4
        elif args.dtype in ["fp8_e5m2", "fp8_e4m3"]:
            dtype_size = 1

        head_num_per_tp = max(head_num // tp_size, 1)
        cell_size = head_num_per_tp * head_dim * layer_num * 2 * dtype_size
        rest_memory = available_gpu_memory * args.mem_fraction_static * (1024**3)

        if args.max_total_num_tokens is None:
            max_total_num_tokens = int(rest_memory // cell_size)
            max_total_num_tokens = (
                max_total_num_tokens // args.page_size
            ) * args.page_size
            logger.info(
                "Calculated max_total_num_tokens=%s from available memory %.2fGB",
                max_total_num_tokens,
                available_gpu_memory,
            )
        else:
            max_total_num_tokens = args.max_total_num_tokens

        if args.max_num_reqs is None:
            max_num_reqs = min(
                max(int(max_total_num_tokens / max_context_len * 512), 2048), 4096
            )
        else:
            max_num_reqs = args.max_num_reqs
    else:
        if (
            args.max_total_num_tokens is None
            or args.head_num is None
            or args.head_dim is None
            or args.layer_num is None
        ):
            raise ValueError(
                "If --model-path is not provided, --max-total-num-tokens, --head-num, "
                "--head-dim, and --layer-num must be provided."
            )
        head_num = args.head_num
        head_dim = args.head_dim
        layer_num = args.layer_num
        max_total_num_tokens = args.max_total_num_tokens
        max_context_len = (
            args.max_context_len if args.max_context_len is not None else 100000
        )
        if args.max_num_reqs is None:
            max_num_reqs = min(
                max(int(max_total_num_tokens / max_context_len * 512), 2048), 4096
            )
        else:
            max_num_reqs = args.max_num_reqs

    server = KVCacheServer(
        socket_path=args.socket_path,
        gpu_ids=gpu_ids,
        max_total_num_tokens=max_total_num_tokens,
        page_size=args.page_size,
        dtype=dtype_map.get(args.dtype, torch.bfloat16),
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        max_num_reqs=max_num_reqs,
        max_context_len=max_context_len,
    )
    server.allocate()
    server.run()
