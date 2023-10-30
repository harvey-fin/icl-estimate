import numpy as np
import click
import pickle
import random
import sys
import torch
import os
import matplotlib.pyplot as plt
import utils
from transformers import set_seed
from model import OPTForSeq2Seq, ConfidenceCalibrator
from metrics import Seq2SeqScorer
from datasets import load_dataset
from promptsource.templates import DatasetTemplates
from sklearn.linear_model import LinearRegression
from utils import load_all_crossfit, load_templates, TokenLengthException
from caching import TaskList
from datasets import DatasetDict
from config import ontology
from typing import *

def run_single(
        model, 
        dataset, 
        dataset_name,
        template, 
        num_shots,
        confidence_metric,
        scorer_function,
        random_seed=1,
        sampling=False):
    set_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    test_prompts, labels = utils.CBQATemplate(dataset=dataset, num_shots=num_shots, template=template).get_prompts()

    outputs, confidences, embeddings = model(test_prompts, confidence_metric=confidence_metric)
    f1_scorer = Seq2SeqScorer(score_metric="spearman", answer_checker=scorer_function)
    f1_results = f1_scorer.score(predictions=outputs, confidences=confidences, labels=labels)

    return confidences, f1_results["score1"], embeddings


@click.command()
@click.option("--model_size", default="opt-125m")
@click.option("--num_shots", default=4)
@click.option("--seed", default=1)
def main(model_size, 
        num_shots, 
        seed):
    scorer_function = "f1"
    confidence_metric = "greedy_loglikelihood"
    max_token = 16
    pkl_path = "pkls_opt_cbqa"

    model = OPTForSeq2Seq(f"facebook/{model_size}", batch_size=None, max_new_tokens=max_token)
    tasks = TaskList.load_cbqa()
    datasets = DatasetDict({task.name: task.load() for task in tasks})
    os.makedirs(pkl_path, exist_ok=True)
    for dataset_name, dataset in datasets.items():
        templates = load_templates(dataset_name, default_only=True)
        for template_name, template in templates.items():
            print(f"*** Running inference with params: ***")
            print(f"OPT {model_size} on {dataset_name} | {num_shots} shots | seed{seed}")
            try:
                pickle_name = f"{pkl_path}/{dataset_name}_{template_name}_{num_shots}shot_seed{seed}_{model_size}.pkl"
                if os.path.exists(pickle_name):
                    print("Inference exists, skipping!")
                    continue

                conf, accs, embed = run_single(model=model,
                    dataset_name=dataset_name,
                    dataset=dataset,
                    template=template,
                    num_shots=num_shots,
                    confidence_metric=confidence_metric,
                    scorer_function=scorer_function,
                    random_seed=seed)
                print("Average f1:", np.mean(accs))
                return_list = {"confs": conf, "accs": accs, "embeds": embed}
                with open(pickle_name, "wb") as f:
                    pickle.dump(return_list, f)
            except TokenLengthException as e:
                print("Token Length Error on", dataset_name, "skipping!")
                pass
            except Exception as e:
                raise
                torch.cuda.empty_cache()
                print("Error on", dataset_name, "skipping!")

if __name__ == "__main__":
    main()
