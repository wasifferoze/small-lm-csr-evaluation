import json
from tqdm import trange
import argparse

from litgpt.api import LLM
from litgpt.api import benchmark_dict_to_markdown_table
from pprint import pprint

parser = argparse.ArgumentParser(description="Benchmark memory usage of the model.")
parser.add_argument('--quantize', type=str, required=True, choices=["none", "bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"], help="The quantize value to use.")
parser.add_argument('--precision', type=str, choices=["16-true", "bf16-true",], help="The precision setting to use")
args = parser.parse_args()

quantize = args.quantize
precision = args.precision
model_name = "phi-35-instruct"

llm_dir = "./checkpoints/microsoft/Phi-3.5-mini-instruct"

llm = LLM.load(llm_dir, distribute=None)
if quantize == "none":
    llm.distribute(fixed_kv_cache_size=500)
else:
    llm.distribute(precision=precision, quantize=quantize, fixed_kv_cache_size=500)

text, bench_d = llm.benchmark(num_iterations=10, prompt="What do llamas eat?", top_k=1, stream=True)

#print(text)
#pprint(bench_d)

bench_d_list = {key: value[1:] for key, value in bench_d.items()}

with open(f"out/benchmark_{model_name}_{precision}_{quantize}.md", "w") as file:
    file.write(benchmark_dict_to_markdown_table(bench_d_list))