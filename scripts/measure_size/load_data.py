from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Mistral3ForConditionalGeneration
import json
import csv
import pandas as pd
import hashlib
import torch
import bitsandbytes
import tabulate
import numpy as np
import sys

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 10000)

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)


qwen8b = "Qwen/Qwen3-8B"

model = AutoModelForCausalLM.from_pretrained(qwen8b, device_map="auto", quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(qwen8b)


import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # repo root
# Override with env vars (e.g. mounted volumes in a container)
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))

sentence = pd.read_csv(DATA_DIR / 'full/sentence_pairs_mask.csv')
sentences = sentence.to_csv(sep="\t", index = False)

df = pd.read_csv(DATA_DIR / 'full/acts_only.csv')     
acts_csv = df.to_csv(sep="\t", index = False)



concept = pd.read_csv(DATA_DIR / 'full/top_n_concepts_from_compexp.csv', sep='\t', quoting=csv.QUOTE_NONE)
concepts = concept.to_csv(sep="\t", index = False)

#for col_name in df.columns:

        #{concepts} USE ONLY THESE SPECIFIC CONCEPTS. NO VARIATIONS OF THEM.
system_prompt = f"""You will be given two aligned tables:
         1. SENTENCES: one premise-hypothesis pair per row, identified by pair_id.
         2. ACTIVATIONS: a single column holding one neuron's binary activations, one row per pair_id. The column header is the neuron's name. The value in row i is the neuron's activation on pair i (1 = the neuron fires, 0 = it does not).

         Your job is to find a logical formula over primitive concepts whose truth values across the pairs best match this neuron's firing pattern.

         Primitive concepts:
        - pre:tok:<word> / hyp:tok:<word>: the word appears in the premise / hypothesis.
        - pre:tag:<POS> / hyp:tag:<POS>: a word with that part-of-speech tag appears in the premise / hypothesis.
        
        Operators: AND, OR, NOT. Use parentheses so precedence is never ambiguous. Use AT MOST 5 primitive concepts in the formula.

        A good formula is true on the pairs where the neuron fires and false on the pairs where it does not.

        Output format: output exactly one block, using the neuron name from the ACTIVATIONS column header:
        <neuron>NEURON_NAME</neuron>
        <formula>FORMULA</formula>

        Example:
        <neuron>neuron_12</neuron>
        <formula>((hyp:tok:nobody OR hyp:tok:no) AND NOT pre:tok:no)</formula>
        Do not write any code. No analysis or justification is necessary. Formula length of 5 at most. 
        """

System_prompt2 ="""The first value corresponds to the column name. All the following values are binary activation values. Tell me how many activating values there are. Respond only with an integer"""


        #Do not  output anything outside these blocks.
        #acts = df[0].to_csv(sep="\t")
        #user_prompt = f"""

        

        #"""
        #SENTENCES:{sentences}
        #ACTIVATIONS:{acts}
        #NEURON_NAME:{col_name}

        #system_prompt = "This is a context test. Explain the data you see passed in"
        #user_prompt = f"{concepts.to_markdown()}"
           

    # messages = [
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt},
    #     ]
col = df.columns[0]

#plain pandas
acts = df[[col]]

#csv
acts_csv = acts.to_csv(sep="\t", index=False)  

#print(acts.shape())
#list form (as string)
acts_list = str(df[col].tolist())

#torch tensor
acts_tensor = torch.tensor(df[col].to_numpy()) 
torch.set_printoptions(profile="full")

#numpy vector
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
acts_np = str(df[col].to_numpy() )


vec = df[col]
n_fire = int((vec != 0).sum())

messages = [
    {"role": "system", "content": acts_np},
    {"role": "user", "content": f"ACTIVATIONS:\n{acts_tensor}"},
]

message1 = [{"role": "user", "content": concepts}]

inputs = tokenizer.apply_chat_template(
            messages, 
            #add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )


        # tokenized prompt output vector
input_ids = inputs["input_ids"]


print("tensor shape(batch size, sequence_length):", input_ids.shape)

print("Num tokens:", input_ids.shape[1])


bytes_input_ids = input_ids.numel() * input_ids.element_size()
mb_input_ids = bytes_input_ids / (1024 ** 2)

print(f"Size in RAM: {mb_input_ids:.6f} MB")

print(f"Number of activations{n_fire}")

MAX_NEW_TOKENS = 100
outputs = model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS)
output_text =tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
print(output_text)