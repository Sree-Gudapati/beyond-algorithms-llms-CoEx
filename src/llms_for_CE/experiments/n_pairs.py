from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Mistral3ForConditionalGeneration
import json
import csv
import pandas as pd
import hashlib
import torch
import bitsandbytes
import tabulate
from llms_for_CE.runs import start_run
import re


# these flags prevent numpy from truncating
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1024)


N_ROWS=100

#models (mainly using qwen3-8b/14b)
qwen8b = "Qwen/Qwen3-8B"
qwen14b = "Qwen/Qwen3-14B"

gemma7b = "google/gemma-7b-it"

phi3miniinstruct = "microsoft/Phi-3-mini-128k-instruct"
phi415bil = "microsoft/phi-4"

mistral7b = "mistralai/Mistral-7B-Instruct-v0.3"
mistral14b ="mistralai/Ministral-3-14B-Instruct-2512"

llama8b = "unsloth/Llama-3.1-8B-Instruct"


#to load mistral models
#from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

#tokenizer = MistralCommonBackend.from_pretrained(mistral14b)
#model = Mistral3ForConditionalGeneration.from_pretrained(mistral14b, device_map="auto", quantization_config=bnb_config)

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(qwen14b)
model = AutoModelForCausalLM.from_pretrained(qwen14b, device_map="auto", quantization_config=bnb_config)

################## Claude Gen ###################
#Just took my code and made it more 'dynamic' 
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]  # repo root
# Override with env vars (e.g. mounted volumes in a container)
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)



sentence = pd.read_csv(DATA_DIR / 'full/sentence_pairs_mask.csv', nrows=N_ROWS)
acts_csv = pd.read_csv(DATA_DIR / 'full/acts_only.csv', nrows=N_ROWS)     
concept = pd.read_csv(DATA_DIR / 'full/top_n_concepts_from_compexp.csv', sep='\t', quoting=csv.QUOTE_NONE)
##################################################
print(sentence.shape)
print(acts_csv.shape)


concepts = concept.to_csv(sep="\t", index = False)
sentences = sentence.to_csv(sep="\t", index = False)

sentences_len = len(sentences.splitlines())   # total lines, including the header row


# write out results to a new timestamped run directory (results.csv inside), and log the run in results/runs.csv
out_file, csv_writer = start_run(RESULTS_DIR, __file__, model=qwen8b, name_params=["n_pairs"],
                                 n_pairs=N_ROWS)



for col_name in acts_csv.columns:

    acts = acts_csv[col_name]

    #putting everything into one table 
    table = sentence.copy()
    table["fires"] = acts.values
    table.index.name = "pair_id"
    sentences_with_acts = table.to_csv(sep="\t")


    #print(sentences_with_acts.shape)

    #skip no activation
    if (acts == 0).all():
        print(f"<neuron>{col_name}</neuron>\n<formula>NO ACTIVATION</formula>", flush=True)
        csv_writer.writerow([col_name, "NO ACTIVATION"])
        continue

        #        Operators: AND, OR, NOT. Use parentheses so precedence is never ambiguous. Use AT MOST 5 primitive concepts in the formula.

    system_prompt = f"""You will be given two tables:
        SENTENCES: one premise-hypothesis pair per row, identified by pair_id.
        ACTIVATIONS: a single column holding one neuron's binary activations, one row per pair_id. The column header is the neuron's name. The value in row i is the neuron's activation on pair i (1 = the neuron fires, 0 = it does not).

         Your job is to incrementally find a logical formula over primitive concepts (one at a time)whose truth values across the pairs best match this neuron's firing pattern.

         Primitive concepts:
        - pre:tok:<word> / hyp:tok:<word>: the word appears in the premise / hypothesis.
        - pre:tag:<POS> / hyp:tag:<POS>: a word with that part-of-speech tag appears in the premise / hypothesis.
        # USE ONLY THE SPECIFIC CONCEPTS THAT APPEAR HERE {concepts} 


        A good formula is true on the pairs where the neuron fires and false on the pairs where it does not.

        Output format: output exactly one block, using the neuron name from the ACTIVATIONS column header:
        <neuron>NEURON_NAME</neuron>
        <formula>FORMULA</formula>

        There should not be any tautologies. The primary goal is to maximize IoU. Parenthesis should be applied strictly to preserve operator precedence in a manner that maximizes IoU. 

        Example:
        <neuron>neuron_12</neuron>
        <formula>((hyp:tok:nobody OR hyp:tok:no) AND NOT pre:tok:no)</formula>
        Do not write any code. No analysis or justification is necessary. Formula length of 5 at most. Make sure to trim and tautologies
        Output only the neuron number formula no analysis. Revise and remove tautologies. Pay close attention to operator precedence: always use explicit parentheses, and double-check they are balanced.
        """

    user_prompt = f"""NEURON_NAME: {col_name}
            SENTENCES: {sentences}
            ACTIVATIONS: {acts}

            Use AT MOST 5 primitive concepts. Maximize IoU.."""

    # run inference
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=4000, do_sample=False)
    output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    #grab only the formulas ignoring any other model output
    m = re.search(r"<formula>(.*?)</formula>", output_text, re.S)
    formula = "PARSE FAILED"
    if m:
        formula = m.group(1).strip()

    del inputs, outputs
    torch.cuda.empty_cache()

    print(f"<neuron>{col_name}</neuron>\n<formula>{formula}</formula>", flush=True)

    csv_writer.writerow([col_name, formula])
    out_file.flush()

out_file.close()