# small-lm-csr-evaluation

This repository contains the code for the paper titled 'Reading Between the Lines: Commonsense Reasoning in Small Language Models,' accepted at 2024 the 10th International Conference on Computer and Communications ( [ICCC](https://www.iccc.org/) ).

## Overview

This repository contains the code for evaluating small language models on Commonsense Reasoning (CSR) tasks. The code is organized into several modules, each responsible for different aspects of the evaluation process.

## Installation

To install the required dependencies, run:
```bash
pip install 'litgpt[all]'
```

## Usage

To run the evaluation, use the following command:
```bash
./run_my_evaluate.sh
```

To run the benchmarks for memory usage and token generation speed, use the following command:
```bash
./run_benchmark.sh
```

## Special cases

If your machine does not have access to internet, download model weights manually from Hugging Face (HF). Then convert these weights to lightning format using:
```bash
python convert_checkpoint.py
```
