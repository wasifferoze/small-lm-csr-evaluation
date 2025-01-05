"""
this will convert record dataset to mcqa format
"""
import json
from tqdm import tqdm

from datasets import load_dataset

record = load_dataset("super_glue", "record")


def preprocess(example):
    """ 
    Will clean passage and query from @highlight and @placeholder
    instead will use - and _ respectively
    """
    pass


def convert_mcqa(example):
    instruction = "In this task, you will be presented with a passage and a question. Identify the best entity to fill in the blank \"_\" based on the passage, and classify the answer by selecting from the provided options. Provide only the selected option.\n\nExample:\nPassage: \"John went to the store to buy some apples. He saw Sarah there and they talked for a while.\"\nQuestion: \"Who went to the store?\"\nOptions: (A) John (B) Sarah (C) Apples\nAnswer: (A) John"
    passage = example["passage"].replace("\n@highlight\n", " - ")
    query = example["query"].replace("@placeholder", "_")
    entities = example["entities"]
    no_options = len(entities)
    # print(no_options)
    options_char = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
                    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "I", "II", "III"]
    options = []
    for i in range(no_options):
        options.append(f"({options_char[i]}) {entities[i]}")

    new_example = {
        "idx": example["idx"]["query"],
        "instruction": instruction,
        "input": f"Passage: {passage}\nQuestion: {query}\nOptions: {' '.join(options)}",
        "answer": example["answers"]
    }
    return new_example

exs = []
for i in tqdm(range(len(record['validation']))):
    ex = convert_mcqa(record['validation'][i])
    exs.append(ex)
    
with open(f"record_val_mcqa_inst.json", "w") as f:
    json.dump(exs, f)
