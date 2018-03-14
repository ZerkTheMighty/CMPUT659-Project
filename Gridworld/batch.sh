#!/bin/bash
echo "Running epsilon batch experiment..."
Array=(0.01 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.60)
for value in ${Array[*]}
do
python grid_sweep_exp.py -e $value
done
