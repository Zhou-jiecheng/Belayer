#!/usr/bin/env bash
# Setup script for the Volcengine ECS "seed" instance.
# Run this ONCE on a fresh ECS to install Docker, swe_exec_server, and pull
# all SWE-Bench images. Then snapshot the ECS into a custom image.
#
# Prerequisites:
#   - A fresh Ubuntu 22.04 ECS instance with >= 1TB disk
#   - train.jsonl copied to ~/train.jsonl on this ECS
#   - This script + swe_exec_server.py + pull_swe_images.sh copied to ~/
#   - Belayer cloned with submodules, or docker-full-checkpoint copied to
#     ~/docker-full-checkpoint/
#
# Usage:
#   bash setup_ecs_seed.sh

set -euo pipefail

echo "========================================"
echo "SWE ECS Seed Setup — $(date)"
echo "========================================"

# ── 1. Install Docker ─────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "[1/5] Installing Docker..."
    curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | apt-key add -
    add-apt-repository "deb [arch=amd64] https://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io
    systemctl enable docker
    systemctl start docker
    echo "Docker installed: $(docker --version)"
else
    echo "[1/5] Docker already installed: $(docker --version)"
fi

# ── 2. Install checkpoint runtime and Python dependencies ─────────────
echo "[2/5] Installing Python/CRIU dependencies and docker-full-checkpoint..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip criu > /dev/null
pip3 install flask --quiet

DFC_SOURCE=""
for candidate in \
    "$(dirname "$0")/../../docker-full-checkpoint" \
    "${HOME}/docker-full-checkpoint" \
    "$(dirname "$0")/../../../docker-full-checkpoint"; do
    if [ -f "${candidate}/pyproject.toml" ]; then
        DFC_SOURCE="${candidate}"
        break
    fi
done
if [ -z "${DFC_SOURCE}" ]; then
    echo "ERROR: docker-full-checkpoint not found. Copy it to ~/docker-full-checkpoint first."
    exit 1
fi
mkdir -p /opt/docker-full-checkpoint
cp -a "${DFC_SOURCE}/." /opt/docker-full-checkpoint/
pip3 install --no-deps /opt/docker-full-checkpoint --quiet

# Docker checkpoint/restore requires daemon experimental mode and the legacy
# overlay2 metadata layout (containerd-snapshotter disabled).
mkdir -p /etc/docker
python3 - <<'PY'
import json
from pathlib import Path

path = Path("/etc/docker/daemon.json")
payload = {}
if path.exists():
    payload = json.loads(path.read_text(encoding="utf-8") or "{}")
payload["experimental"] = True
features = payload.get("features")
if not isinstance(features, dict):
    features = {}
features["containerd-snapshotter"] = False
payload["features"] = features
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
systemctl restart docker
docker info --format 'experimental={{.ExperimentalBuild}} driver={{.Driver}} root={{.DockerRootDir}}'
criu check

# ── 3. Install swe_exec_server as a systemd service ──────────────────
echo "[3/5] Setting up swe_exec_server..."
mkdir -p /opt/swe-exec-server
mkdir -p /var/lib/swe-checkpoints /dev/shm/docker-full-checkpoint

if [ -f ~/swe_exec_server.py ]; then
    cp ~/swe_exec_server.py /opt/swe-exec-server/server.py
elif [ -f "$(dirname "$0")/swe_exec_server.py" ]; then
    cp "$(dirname "$0")/swe_exec_server.py" /opt/swe-exec-server/server.py
else
    echo "ERROR: swe_exec_server.py not found. Copy it to ~/ first."
    exit 1
fi

if [ -f ~/container_pool_config.json ]; then
    cp ~/container_pool_config.json /opt/swe-exec-server/container_pool_config.json
elif [ -f "$(dirname "$0")/container_pool_config.json" ]; then
    cp "$(dirname "$0")/container_pool_config.json" /opt/swe-exec-server/container_pool_config.json
fi

cat > /etc/systemd/system/swe-exec-server.service <<'EOF'
[Unit]
Description=SWE Docker Exec Server
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=simple
Environment=SWE_CHECKPOINT_BACKEND=full
Environment=SWE_CHECKPOINT_DIR=/var/lib/swe-checkpoints
Environment=SWE_FULL_CHECKPOINT_STATE_ROOT=/var/lib/swe-checkpoints/full-checkpoint-state
Environment=SWE_FULL_CHECKPOINT_PROJECT_ROOT=/opt/docker-full-checkpoint
Environment=SWE_FULL_CHECKPOINT_RUNTIME_STAGING_ROOT=/dev/shm/docker-full-checkpoint
Environment=SWE_CHECKPOINT_MAX_INFLIGHT=1
ExecStart=/usr/bin/python3 /opt/swe-exec-server/server.py --port 5000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable swe-exec-server
systemctl start swe-exec-server

for i in {1..10}; do
    if curl -fsS http://localhost:5000/healthz >/dev/null 2>&1; then
        echo "swe_exec_server is running on :5000"
        break
    fi
    sleep 1
done

# ── 4. Validate full checkpoint capability ───────────────────────────
echo "[4/5] Validating Docker checkpoint capability..."
docker checkpoint --help >/dev/null
docker start --help | grep -q -- --checkpoint
findmnt -no PROPAGATION /sys/fs/cgroup | grep -Eq '^(private|rprivate)$'
curl -fsS http://localhost:5000/healthz | python3 -m json.tool

# ── 5. Pull SWE-Bench Docker images ──────────────────────────────────
TRAIN=${TRAIN:-${HOME}/train.jsonl}
if [ ! -f "${TRAIN}" ]; then
    echo "[5/5] SKIPPED: ${TRAIN} not found."
    echo "  Copy train.jsonl to ${TRAIN} and run:"
    echo "    TRAIN=${TRAIN} bash ~/pull_swe_images.sh"
else
    echo "[5/5] Pulling SWE-Bench Docker images from ${TRAIN}..."
    PULL_SCRIPT=""
    if [ -f ~/pull_swe_images.sh ]; then
        PULL_SCRIPT=~/pull_swe_images.sh
    elif [ -f "$(dirname "$0")/pull_swe_images.sh" ]; then
        PULL_SCRIPT="$(dirname "$0")/pull_swe_images.sh"
    fi

    if [ -n "${PULL_SCRIPT}" ]; then
        LOG_DIR=/var/log/swe-images TRAIN="${TRAIN}" bash "${PULL_SCRIPT}"
    else
        echo "  WARNING: pull_swe_images.sh not found, skipping image pull."
        echo "  Copy it and run manually."
    fi
fi

echo ""
echo "========================================"
echo "Setup complete — $(date)"
echo ""
echo "Next steps:"
echo "  1. Verify: curl http://localhost:5000/healthz"
echo "  2. Check images: curl http://localhost:5000/images | python3 -m json.tool | head"
echo "  3. Stop this ECS in Volcengine console"
echo "  4. Create a custom image from this ECS"
echo "  5. Use that image_id in your training script"
echo "========================================"
