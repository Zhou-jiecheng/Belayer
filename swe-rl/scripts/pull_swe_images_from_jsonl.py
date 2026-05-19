#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
# xingyaoww/sweb.eval.x86_64.getmoto_s_moto-4990:latest
# registry.h.pjlab.org.cn/ailab-sys-sys_gpu/swe-rl:getmoto_s_moto-4990
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read docker_name from JSONL metadata.instance.instance_id, "
            "find local docker images whose REPOSITORY contains "
            "'xingyaoww/{docker_name}', then tag/push to target registry."
        )
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("data/train.jsonl"),
        help="Path to JSONL file (default: data/train.jsonl)",
    )
    parser.add_argument(
        "--source-namespace",
        default="xingyaoww",
        help="Source repository namespace to match (default: xingyaoww)",
    )
    parser.add_argument(
        "--source-image-prefix",
        default="sweb.eval.x86_64.",
        help="Source image name prefix right after namespace (default: sweb.eval.x86_64.)",
    )
    parser.add_argument(
        "--target-repo",
        default="registry.h.pjlab.org.cn/ailab-sys-sys_gpu/swe-rl",
        help="Target docker repository (without tag)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when one docker tag/push step fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print docker commands without executing them",
    )
    return parser.parse_args()


def extract_docker_name(record: dict[str, Any]) -> str | None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    instance = metadata.get("instance")
    if not isinstance(instance, dict):
        return None
    instance_id = instance.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id.strip():
        return None
    # Keep consistent with existing image naming in swe-rl.
    return instance_id.strip().replace("__", "_s_")


def load_docker_names(jsonl_path: Path) -> tuple[list[str], int, int]:
    docker_names: list[str] = []
    seen: set[str] = set()
    processed = 0
    skipped = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                skipped += 1
                continue

            processed += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Line {line_num}: invalid JSON, skipped ({exc})")
                skipped += 1
                continue

            docker_name = extract_docker_name(item)
            if docker_name is None:
                print(f"[WARN] Line {line_num}: missing metadata.instance.instance_id, skipped")
                skipped += 1
                continue

            if docker_name not in seen:
                seen.add(docker_name)
                docker_names.append(docker_name)

    return docker_names, processed, skipped


def list_local_images() -> list[dict[str, str]]:
    cmd = ["docker", "images", "--format", "{{json .}}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"docker images failed (exit={result.returncode}): {result.stderr.strip()}"
        )

    images: list[dict[str, str]] = []
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        repo = str(row.get("Repository", "") or "")
        tag = str(row.get("Tag", "") or "")
        image_id = str(row.get("ID", "") or "")
        if not repo:
            continue
        images.append({"repository": repo, "tag": tag, "id": image_id})
    return images


def source_ref_from_image(row: dict[str, str]) -> str:
    repo = row["repository"]
    tag = row["tag"]
    image_id = row["id"]
    if tag and tag != "<none>":
        return f"{repo}:{tag}"
    if image_id:
        return image_id
    return repo


def run_cmd(cmd: list[str], dry_run: bool) -> int:
    print(f"[CMD] {' '.join(cmd)}")
    if dry_run:
        return 0
    result = subprocess.run(cmd)
    return result.returncode


def main() -> int:
    args = parse_args()
    jsonl_path: Path = args.jsonl

    if not jsonl_path.exists():
        print(f"[ERROR] JSONL file not found: {jsonl_path}")
        return 1

    docker_names, processed, skipped = load_docker_names(jsonl_path)
    if not docker_names:
        print(f"[SUMMARY] processed={processed}, skipped={skipped}, docker_names=0, nothing to do")
        return 0

    try:
        local_images = list_local_images()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"[INFO] Local docker images ({len(local_images)}):")
    for idx, row in enumerate(local_images, start=1):
        print(f"  {idx:4d}. {row['repository']}:{row['tag']} ({row['id']})")

    succeeded = 0
    failed = 0
    missing = 0

    for i, docker_name in enumerate(docker_names, start=1):
        primary_needle = f"{args.source_namespace}/{args.source_image_prefix}{docker_name}"
        fallback_needle = f"{args.source_namespace}/{docker_name}"
        matches = [
            row
            for row in local_images
            if primary_needle in row["repository"] or fallback_needle in row["repository"]
        ]

        print(
            f"\n[{i}/{len(docker_names)}] docker_name={docker_name} "
            f"primary_needle={primary_needle} fallback_needle={fallback_needle}"
        )
        if not matches:
            print("[WARN] No local image matched this docker_name")
            missing += 1
            continue

        if len(matches) > 1:
            refs = [source_ref_from_image(row) for row in matches]
            print(f"[WARN] Multiple matches found, will use the first one: {refs}")

        source_ref = source_ref_from_image(matches[0])
        target_ref = f"{args.target_repo}:{docker_name}"

        rc = run_cmd(["docker", "tag", source_ref, target_ref], dry_run=args.dry_run)
        if rc != 0:
            failed += 1
            print(f"[ERROR] docker tag failed for source={source_ref} target={target_ref}")
            if args.stop_on_error:
                return rc
            continue

        rc = run_cmd(["docker", "push", target_ref], dry_run=args.dry_run)
        if rc != 0:
            failed += 1
            print(f"[ERROR] docker push failed for target={target_ref}")
            if args.stop_on_error:
                return rc
            continue

        succeeded += 1

    print(
        "\n[SUMMARY] "
        f"processed={processed}, skipped={skipped}, unique_docker_names={len(docker_names)}, "
        f"succeeded={succeeded}, missing={missing}, failed={failed}, dry_run={args.dry_run}"
    )


    if succeeded == 0:
        print("[WARN] No docker images were successfully tagged/pushed.")
        print("start pull dockers from registry: ")
        for docker_name in docker_names:
            target_ref = f"{args.target_repo}:{docker_name}"
            rc = run_cmd(["docker", "pull", target_ref], dry_run=args.dry_run)
            if rc != 0:
                failed += 1
                print(f"[ERROR] docker pull failed for target={target_ref}")
                if args.stop_on_error:
                    return rc
                continue
            succeeded += 1
            print(f"[INFO] Successfully pulled {target_ref} [{succeeded} / {len(docker_names)}]")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
