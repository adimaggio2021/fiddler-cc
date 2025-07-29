#!/bin/bash

echo "Running sweep"
echo ""

for hit_rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    echo "Running latency with hit_rate = $hit_rate"
    python3.10 latency.py --beam_width 1 --cpu-offload 0 --mem_portion 0.5 --hit_rate $hit_rate
done
