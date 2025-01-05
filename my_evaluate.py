import os
import json
from tqdm import trange
import argparse

from litgpt.prompts import Alpaca, Phi3
from litgpt import LLM

def any_type(value):
    return value

parser = argparse.ArgumentParser(description="Evaluate datasets with different quantize values and precision settings.")
parser.add_argument('--dataset', type=str, required=True, choices=["record", "cosmosqa"], help="The dataset to evaluate.")
parser.add_argument('--quantize', type=str, required=True, choices=["none", "bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"], help="The quantize value to use.")
parser.add_argument('--precision', type=str, choices=["16-true", "bf16-true",], help="The precision setting to use")
args = parser.parse_args()


record_val_file = "./data/record_val_mcqa_inst.json"
cosmosqa_val_file = "./data/cosmosqa_val_mcqa_inst.json"

dataset_name = args.dataset
quantize = args.quantize
precision = args.precision
model_name = "phi-35-instruct"
split_name = "val"

if dataset_name == "record":
    ds_file = record_val_file
elif dataset_name == "cosmosqa":
    ds_file = cosmosqa_val_file
else:
    raise NotImplementedError("This dataset is not implemented yet.")

# Load the test data
with open(ds_file, "r") as file:
    test_data = json.load(file)

# Print the first example
# print(test_data[0])

# test_data = test_data[7372:]
# # print(len(test_data))
# # print(test_data[0])

# system prompts for the model
alpaca_style = Alpaca()
prompt_style = Phi3()


llm_dir = "./checkpoints/microsoft/Phi-3.5-mini-instruct"
# create an output directory if it doesn't exist
if quantize == "none":
    output_dir = "./out"
else:
    output_dir = "./out/quant"
os.makedirs(output_dir, exist_ok=True)


llm = LLM.load(llm_dir)
if quantize != "none":
    llm.distribute(precision=precision, quantize=quantize)

output_file_path = f"{output_dir}/output_{dataset_name}_{split_name}_{model_name}_precs_{precision}_quant_{quantize}.jsonl"
with open(output_file_path, "a") as file:
    for i, data in enumerate(test_data):
        idx = data["idx"]
        instruction = data["instruction"]
        input_text = data["input"]
        true_answer = data["answer"]
        output_text = f"\nAnswer: "
        prompt = f"{instruction}\n\n{input_text}{output_text}"
        try:
            response = llm.generate(prompt_style.apply(prompt))
            file.write(json.dumps({
                "idx": idx,
                "true_answer": true_answer,
                "response": response
            }, indent=4) + "\n")
        except Exception as e:
            print(f"Error processing data index {idx}: {e}")

print(f"Output written to {output_file_path}")