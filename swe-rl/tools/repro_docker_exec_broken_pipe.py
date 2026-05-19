from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SWE_RL_ROOT = Path(__file__).resolve().parents[1]
SLIME_ROOT = REPO_ROOT / "slime"
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (TOOLS_ROOT, SLIME_ROOT, SWE_RL_ROOT):
    sys.path.insert(0, str(path))

from replay_swe_traj_checkpoint import ReplayEnvClient  # noqa: E402
from swe_utils import get_docker_image_name  # noqa: E402


DEFAULT_LONG_COMMAND = (
    "cd /testbed && pip install boto3 moto mypy-boto3-dynamodb "
    "&& python3 test_trailing_comma.py"
)


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


async def _run_exec(
    client: SweEnvClient,
    *,
    lease_id: str,
    command: str,
    cwd: str,
    timeout: int,
    label: str,
) -> dict:
    started = time.time()
    try:
        out = await client.exec(lease_id, command, cwd=cwd, timeout=timeout)
        return {
            "label": label,
            "ok": True,
            "wall_time_sec": time.time() - started,
            "returncode": out.get("returncode"),
            "output_preview": str(out.get("output", "") or "")[:2000],
        }
    except Exception as exc:
        return {
            "label": label,
            "ok": False,
            "wall_time_sec": time.time() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def _churn_worker(
    worker_idx: int,
    *,
    client: SweEnvClient,
    image: str,
    cwd: str,
    rounds: int,
    exec_timeout: int,
    close_delay_sec: float,
) -> list[dict]:
    events: list[dict] = []
    for round_idx in range(rounds):
        lease_id = None
        try:
            alloc = await client.allocate(image=image, instance_id=f"churn-{worker_idx}-{round_idx}", cwd=cwd)
            lease_id = alloc["lease_id"]
            events.append(
                {
                    "worker_idx": worker_idx,
                    "round_idx": round_idx,
                    "phase": "allocate",
                    "ok": True,
                    "lease_id": lease_id,
                    "container_id": alloc.get("container_id"),
                }
            )
            exec_result = await _run_exec(
                client,
                lease_id=lease_id,
                command="echo churn && pwd",
                cwd=cwd,
                timeout=exec_timeout,
                label=f"churn-{worker_idx}-{round_idx}-exec",
            )
            events.append(
                {
                    "worker_idx": worker_idx,
                    "round_idx": round_idx,
                    "phase": "exec",
                    **exec_result,
                }
            )
            if close_delay_sec > 0:
                await asyncio.sleep(close_delay_sec)
        except Exception as exc:
            events.append(
                {
                    "worker_idx": worker_idx,
                    "round_idx": round_idx,
                    "phase": "worker_error",
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            if lease_id is not None:
                try:
                    await client.close(lease_id)
                    events.append(
                        {
                            "worker_idx": worker_idx,
                            "round_idx": round_idx,
                            "phase": "close",
                            "ok": True,
                            "lease_id": lease_id,
                        }
                    )
                except Exception as exc:
                    events.append(
                        {
                            "worker_idx": worker_idx,
                            "round_idx": round_idx,
                            "phase": "close",
                            "ok": False,
                            "lease_id": lease_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    return events


async def _main_async(args: argparse.Namespace) -> int:
    client = ReplayEnvClient(base_url=args.base_url)
    image = args.image_name or get_docker_image_name(
        {"instance_id": args.instance_id},
        data_source=args.data_source,
    )
    report: dict = {
        "base_url": args.base_url,
        "instance_id": args.instance_id,
        "image_name": image,
        "cwd": args.cwd,
        "victim_command": args.command,
        "victim_rounds": args.victim_rounds,
        "churn_workers": args.churn_workers,
        "churn_rounds": args.churn_rounds,
        "started_at": time.time(),
    }

    victim_alloc = await client.allocate(image=image, instance_id=args.instance_id, cwd=args.cwd)
    victim_lease_id = victim_alloc["lease_id"]
    report["victim_lease"] = victim_alloc

    churn_task = None
    if args.churn_workers > 0 and args.churn_rounds > 0:
        async def _run_churn() -> list[dict]:
            results = await asyncio.gather(
                *[
                    _churn_worker(
                        worker_idx=i,
                        client=client,
                        image=image,
                        cwd=args.cwd,
                        rounds=args.churn_rounds,
                        exec_timeout=args.exec_timeout,
                        close_delay_sec=args.churn_close_delay_sec,
                    )
                    for i in range(args.churn_workers)
                ]
            )
            flattened: list[dict] = []
            for chunk in results:
                flattened.extend(chunk)
            return flattened

        churn_task = asyncio.create_task(_run_churn())

    victim_exec_results: list[dict] = []
    for round_idx in range(args.victim_rounds):
        result = await _run_exec(
            client,
            lease_id=victim_lease_id,
            command=args.command,
            cwd=args.cwd,
            timeout=args.exec_timeout,
            label=f"victim-{round_idx}",
        )
        result["round_idx"] = round_idx
        victim_exec_results.append(result)
        if args.victim_pause_sec > 0 and round_idx + 1 < args.victim_rounds:
            await asyncio.sleep(args.victim_pause_sec)
    report["victim_exec_results"] = victim_exec_results

    if churn_task is not None:
        report["churn_events"] = await churn_task
    else:
        report["churn_events"] = []

    if not args.keep_victim_open:
        try:
            await client.close(victim_lease_id)
            report["victim_closed"] = True
        except Exception as exc:
            report["victim_closed"] = False
            report["victim_close_error"] = f"{type(exc).__name__}: {exc}"
    else:
        report["victim_closed"] = False

    report["finished_at"] = time.time()
    report["wall_time_sec"] = report["finished_at"] - report["started_at"]
    report["victim_failures"] = sum(1 for item in victim_exec_results if not item.get("ok", False))
    report["victim_nonzero_returncodes"] = sum(
        1 for item in victim_exec_results if item.get("ok", False) and int(item.get("returncode") or 0) != 0
    )

    text = json.dumps(report, indent=2, ensure_ascii=False, default=_json_default)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Docker exec broken-pipe errors with one victim long exec plus optional churn leases.",
    )
    parser.add_argument("--base-url", required=True, help="swe_env_pool_server base URL")
    parser.add_argument("--instance-id", default="getmoto__moto-5189", help="SWE instance id used to infer image name")
    parser.add_argument("--data-source", default="swe-gym", help="Data source for docker image naming")
    parser.add_argument("--image-name", default=None, help="Explicit docker image name override")
    parser.add_argument("--cwd", default="/testbed", help="Working directory in the container")
    parser.add_argument("--command", default=DEFAULT_LONG_COMMAND, help="Victim long-running exec command")
    parser.add_argument("--exec-timeout", type=int, default=1800, help="Exec timeout in seconds")
    parser.add_argument("--victim-rounds", type=int, default=1, help="How many long execs to run in the victim container")
    parser.add_argument("--victim-pause-sec", type=float, default=0.0, help="Pause between victim exec rounds")
    parser.add_argument("--churn-workers", type=int, default=8, help="How many concurrent churn workers to run")
    parser.add_argument("--churn-rounds", type=int, default=4, help="How many allocate/exec/close rounds per churn worker")
    parser.add_argument("--churn-close-delay-sec", type=float, default=0.0, help="Delay before closing each churn lease")
    parser.add_argument("--keep-victim-open", action="store_true", help="Leave the victim lease open for manual inspection")
    parser.add_argument("--output-json", default=None, help="Optional JSON output path")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
