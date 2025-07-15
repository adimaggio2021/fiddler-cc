#!/bin/bash

echo "Running sweep"
echo ""

for mem_val in 0.25 0.5 1.0
do
    echo "Running latency with mem_portion = $mem_val"
    python3.10 latency.py --mem_portion $mem_val
done
