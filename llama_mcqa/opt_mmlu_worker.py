import numpy as np
import click
import pickle
import random
import torch
import os
import sys
import fire
import time
import json
import utils
from transformers import set_seed
from typing import Tuple
from opt_model import OPTForMCQA
from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from metrics import LLaMAScorer
from config import ontology
from datasets import Dataset, DatasetDict
from utils import TokenLengthException
from caching import TaskList


"""
function to run OPT model on given setting
template: using mmlu template on crossfit-MCQA tasks
returns:
    confs, accs
"""
def run_opt(model: OPTForMCQA,
        dataset: DatasetDict,
        subject: str,
        num_shots: int,
        checker_function: str,
        confidence_metric: str,
        template: str = "mmlu",
        random_seed: int = 1):

    set_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    if random_seed == 0:
        method = "default"
    else:
        method = "sample"

    test_prompts, labels = utils.LLaMATemplate(dataset=dataset,
            subject=subject,
            num_shots=num_shots,
            template=template,
            method=method).get_prompts()

    outputs, confidences, embeddings = model(examples=test_prompts)
    # use the multiple-choice scorer function
    accs, confs = LLaMAScorer(answer_checker=checker_function,
            confidence_metric=confidence_metric).score(probs=confidences, labels=labels)

    return confs, accs, embeddings


@click.command()
@click.option("--model_size", type=str, default="opt-6.7b")
@click.option("--num_shots", type=int, default=5)
@click.option("--temperature", type=float, default=0)
@click.option("--template", type=str, default="mmlu")
@click.option("--seed", type=int, default=0)
def main(model_size: str, num_shots: int, temperature: float, seed: int, template:str):

    # fix the max token for mmlu to be 5
    max_token=5
    checker_function = "exact_match"
    confidence_metric = "label_normalize"
    model=OPTForMCQA(model_name=f"facebook/{model_size}", max_length=2048, max_new_tokens=max_token)
    tasks = TaskList.load_mmlu()

    print("\n===============================Loading Tasks==============================\n")
    datasets = DatasetDict({task.subset: task.load() for task in tasks})
    pkl_path = "pkls_opt_mmlu/"
    os.makedirs(pkl_path, exist_ok=True)

    # run inference for each dataset
    for dataset_name, dataset in datasets.items():
        subject = " ".join(dataset_name.split("_"))
        print(f"*** Running inference with params: ***")
        print(f"OPT {model_size} | {num_shots} shot | {template} | {seed} seed on {dataset_name}")
        try:
            pickle_name = f"{pkl_path}/{dataset_name}_{template}_{num_shots}shot_seed{seed}_{model_size}.pkl"
            if os.path.exists(pickle_name):
                print("Inference exists, skipping!")
                continue
            
            confs, accs, embeds = run_opt(model=model,
                dataset=dataset,
                subject=subject,
                num_shots=num_shots,
                checker_function=checker_function,
                confidence_metric=confidence_metric,
                random_seed=seed,
                template=template,
                )

            print("Average accs:", np.mean(accs))
            return_list = {"confs": confs, "accs": accs, "embeds": embeds}
            with open(pickle_name, "wb") as f:
                pickle.dump(return_list, f)
        except RuntimeError as e:
            print("Token Length Error on", dataset_name, "skipping!")
            continue
        except Exception as e:
            raise
            torch.cuda.empty_cache()
            print("Error on", dataset_name, "skipping!")

if __name__ == "__main__":
    main()
