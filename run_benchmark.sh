#!/bin/bash

# Datasets and quantize values
quantize_values=("none" "bnb.nf4" "bnb.nf4-dq" "bnb.fp4" "bnb.fp4-dq" "bnb.int8")

# Loop through each quantize value
for quantize in "${quantize_values[@]}"; do
    # Set the precision based on the quantize value
    if [ "$quantize" == "bnb.int8" ]; then
        precision="16-true"
    else
        precision="bf16-true"
    fi

    # Run your Python script with the dataset, quantize value, and precision as arguments
    echo "Running benchmark with quantize: $quantize and precision: $precision"
    python benchmark_generation_memory.py --quantize "$quantize" --precision "$precision"
done
