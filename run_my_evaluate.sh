#!/bin/bash

# Datasets and quantize values
datasets=("record" "cosmosqa")
quantize_values=("none" "bnb.nf4" "bnb.nf4-dq" "bnb.fp4" "bnb.fp4-dq" "bnb.int8")

# Loop through each dataset
for dataset in "${datasets[@]}"; do
  # Loop through each quantize value
  for quantize in "${quantize_values[@]}"; do
    # Set the precision based on the quantize value
    if [ "$quantize" == "bnb.int8" ]; then
      precision="16-true"
    else
      precision="bf16-true"
    fi
    
    # Run your Python script with the dataset, quantize value, and precision as arguments
    echo "Running evaluation for dataset: $dataset with quantize: $quantize and precision: $precision"
    python my_evaluate.py --dataset "$dataset" --quantize "$quantize" --precision "$precision"
  done
done
