#!/bin/bash

LOG_FILE="container_concurrency.log"

echo "开始监控容器数量，日志文件：$LOG_FILE"
echo "按 Ctrl+C 停止监控"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    COUNT=$(docker ps | wc -l)
    # 减去1是为了排除 docker ps 输出的表头行
    CONTAINER_COUNT=$((COUNT - 1))
    echo "${TIMESTAMP} - 运行中的容器数量: ${CONTAINER_COUNT}" | tee -a "$LOG_FILE"
    sleep 2
done