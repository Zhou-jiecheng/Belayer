#!/bin/bash
CONTAINER_ID="323e33833948"
RUNS=5

for i in $(seq 1 $RUNS); do
    echo "Run $i..."
    start=$(date +%s%N)
    docker commit $CONTAINER_ID bench-$i > /dev/null
    end=$(date +%s%N)
    ms=$(( (end - start) / 1000000 ))
    echo "  耗时: ${ms} ms"
    docker rmi bench-$i > /dev/null
done
