import string
import os
import re
import sys
import csv
import random
import pickle
import torch
import pandas as pd
from typing import *
from itertools import chain
from typing import *
from tqdm import tqdm
from datasets import load_dataset, DatasetDict, Dataset
from promptsource.templates import DatasetTemplates


class PromptMinusTestExample:
    """ Prompt formatted with demonstrations, but not with test example """
    def __init__(self, template, train_examples, num_shots=None, sampling='random'):
        if num_shots is not None:
            assert len(train_examples) > num_shots
        self.template = template
        if sampling == 'random':
            assert num_shots is not None
            ic_indices = random.sample(range(len(train_examples)), num_shots)
            ic_examples = train_examples.select(ic_indices)
        elif sampling == 'all':
            ic_examples = train_examples
        else:
            raise ValueError
        self.ic_examples = ic_examples
        self.prompt = ""
        for ic_example in ic_examples:
            self.prompt += template.apply(ic_example) + "\n"
    
    def get_ic_examples(self) -> List[Dict[str, str]]:
        return self.ic_examples

    def apply(self, test_example):
        return self.prompt + self.template.apply(test_example, include_answer=False)


class T5Template:
    """ T5 five shot default template """
    def __init__(self, dataset: DatasetDict, subject: str, num_shots: int =5, template: str = "mmlu", method: str="default"):
        self.ic_prompt: str = ""
        self.template = template
        self.subject = subject
        self.dev = dataset["dev"]
        self.test = dataset["test"]
        self.valid = dataset["validation"]
        self.prompts: List[str] = []
        self.labels: List[int] = self.test["answer"]

        # sample in-context examples from validation set
        if method == "sample" or num_shots!=5:
            self.ic_examples: Dict[str, List[str]] = {}
            for key in self.valid.features.keys():
                self.ic_examples[key] = self.dev[key] + self.valid[key]

            ic_indices = random.sample(range(len(self.ic_examples["question"])), num_shots)
            for key in self.valid.features.keys():
                self.ic_examples[key] = [self.ic_examples[key][i] for i in ic_indices]
        else:
            self.ic_examples = self.dev

        if self.template == "subject":
            self.ic_prompt += "The following are multiple choice questions (with answers) about " + self.subject + "\n\n" 
        # get in-context prompts
        for i, (q, c, a) in enumerate(zip(self.ic_examples["question"], self.ic_examples["choices"], self.ic_examples["answer"])):
            self.ic_prompt = self._single_shot(self.ic_prompt, q, c, a)

        # form test prompts
        for q, c, a in zip(self.test["question"], self.test["choices"], self.test["answer"]):
            self.prompts.append(self._single_shot(self.ic_prompt, q, c, a, include_answer=False))

    def _single_shot(self, prompt: str, q: str, c: List[str], a: int, include_answer: bool =True) -> str:
        labels = ["A", "B", "C", "D"]
            
        prompt += q + "\n"
        for label, choice in zip(labels, c):
            prompt += label + ". " + choice + "\n"
        if include_answer:
            prompt += "\nAnswer:" + labels[a] + ". " + c[a] + "\n\n"
        else:
            prompt += "\nAnswer:"
        return prompt

    def get_prompts(self) -> List[str]:
        return self.prompts, self.labels


class LLaMATemplate:
    """ LLaMA five shot default template """
    def __init__(self, dataset: DatasetDict, subject: str, num_shots: int =5, template: str = "mmlu", method: str="default"):
        self.ic_prompt: str = ""
        self.subject = subject
        self.template = template
        self.dev = dataset["dev"]
        self.test = dataset["test"]
        self.valid = dataset["validation"]
        self.prompts: List[str] = []
        self.labels: List[int] = self.test["answer"]

        # sample in-context examples from validation set
        if method == "sample" or num_shots!=5:
            self.ic_examples: Dict[str, List[str]] = {}
            for key in self.valid.features.keys():
                self.ic_examples[key] = self.dev[key] + self.valid[key]

            ic_indices = random.sample(range(len(self.ic_examples["question"])), num_shots)
            for key in self.valid.features.keys():
                self.ic_examples[key] = [self.ic_examples[key][i] for i in ic_indices]
        else:
            self.ic_examples = self.dev

        if self.template == "subject":
            self.ic_prompt += "The following are multiple choice questions (with answers) about " + self.subject + "\n\n" 
        elif self.template == "gopher":
            self.ic_prompt += "A highly knowledgeable and intelligent AI answers multiple-choice questions about " + self.subject + "\n\n"
        elif self.template == "gpt":
            self.ic_prompt += "Test your knowledge of " + self.subject + " with multiple-choice questions and discover the correct answers.\n\n"
        elif self.template == "user":
            self.ic_prompt += "You are an expert in " + self.subject + ", here are some multiple choice questions\n\n"


        # get in-context prompts
        for i, (q, c, a) in enumerate(zip(self.ic_examples["question"], self.ic_examples["choices"], self.ic_examples["answer"])):
            self.ic_prompt = self._single_shot(self.ic_prompt, q, c, a)

        # form test prompts
        for q, c, a in zip(self.test["question"], self.test["choices"], self.test["answer"]):
            self.prompts.append(self._single_shot(self.ic_prompt, q, c, a, include_answer=False))

    def _single_shot(self, prompt: str, q: str, c: List[str], a: int, include_answer: bool =True) -> str:
        labels = ["(A)", "(B)", "(C)", "(D)"]
        prompt += q + "\n"

        for label, choice in zip(labels, c):
            prompt += label + " " + choice + "\n"
        if include_answer:
            prompt += "\nAnswer:" + labels[a] + " " + c[a] + "\n\n"
        else:
            prompt += "\nAnswer:"
        return prompt

    def get_prompts(self) -> List[str]:
        return self.prompts, self.labels
            

class MCQATemplate:
    """ MCQA n shot default template """
    def __init__(self, dataset: DatasetDict, num_shots: int =5, template: str = "mmlu"):
        self.ic_prompt: str = ""
        self.template = template
        self.train = dataset["train"]
        self.test = dataset["test"]
        self.prompts: List[str] = []
        self.labels: List[int] = self.test["answers"]
        self.label_space: List[str] = []

        for idx in range(len(self.train["choices"][0])):
            self.label_space.append("(" + chr(ord("A")+idx)+")")

        # get in-context prompts
        ic_indices = random.sample(range(len(self.train["answers"])), num_shots)
        self.ic_examples={}
        for key in self.train.keys():
            self.ic_examples[key] = [self.train[key][i] for i in ic_indices]
        for i, (q, c, a) in enumerate(zip(self.ic_examples["questions"], self.ic_examples["choices"], self.ic_examples["answers"])):
            self.ic_prompt = self._single_shot(self.ic_prompt, q, c, a)

        # form test prompts
        for q, c, a in zip(self.test["questions"], self.test["choices"], self.test["answers"]):
            self.prompts.append(self._single_shot(self.ic_prompt, q, c, a, include_answer=False))

    def _single_shot(self, prompt: str, q: str, c: List[str], a: int, include_answer: bool =True) -> str:
        prompt += q + "\n"
        if self.template == "mmlu":
            for label, choice in zip(self.label_space, c):
                prompt += label + " " + choice + "\n"
        if include_answer:
            prompt += "\nAnswer:" + self.label_space[a] + " " + c[a] + "\n\n"
        else:
            prompt += "\nAnswer:"
        return prompt

    def get_prompts(self) -> List[str]:
        return self.prompts, self.labels, self.label_space


class DefaultTemplate:
    def apply(self, test_example, include_answer=True):
        if include_answer:
            return test_example["input"] + "\n" + test_example["output"] + "\n"
        else:
            return test_example["input"] + "\n"

def load_templates(dataset_name, default_only=False):
    dataset_name = dataset_name.replace("_", "/")
    if default_only:
        return {"default": DefaultTemplate()}
    try:
        templates = DatasetTemplates(dataset_name).templates
        print(templates)
        assert len(templates) > 0
        return templates
    except:
        return {"default": DefaultTemplate}

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]

def choice_to_label(c):
    """ transform answer choices to labels """
    if c.upper() == "A":
        return 0
    elif c.upper() == "B":
        return 1
    elif c.upper() == "C":
        return 2
    elif c.upper() == "D":
        return 3
    else:
        return 4

def normalize(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

def ppl(text):
    assert model is not None
    assert tokenizer is not None
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    input_ids = input_ids.to("cuda")
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    sentence_prob = loss.item()
    return np.exp(sentence_prob)

class NoPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def load_crossfit(dataset_name, num_test):
    """
    load data from crossfit
    """
    NUM_SHOTS = 32
    SEED = 1
    data = DatasetDict()
    for phase in ["train", "dev", "test"]:
        data_path = os.path.join("data", dataset_name, f"{dataset_name}_{NUM_SHOTS}_{SEED}_{phase}.tsv")
        df = pd.read_csv(data_path, delimiter="\t", names=["input", "output"], quoting=csv.QUOTE_NONE)
        if phase == "test":
            df = df[:num_test]
        dataset = Dataset.from_pandas(df)
        data[phase] = dataset
    return data


def load_all_crossfit(num_test):
    task_list = os.listdir("data")
    datasets = DatasetDict()
    print("Loading tasks...")
    for task in tqdm(task_list):
        dataset = load_crossfit(task, num_test)
        datasets[task] = dataset
    return datasets


def get_prompt(instruction, prompt_template, train_questions, train_answers, num_shots):
    """ OLD METHOD """
    prompt = instruction
    assert len(train_questions) >= num_shots
    ic_indices = random.sample(range(len(train_questions)), num_shots)
    ic_questions = [q for i, q in enumerate(train_questions) if i in ic_indices]
    ic_answers = [a for i, a in enumerate(train_answers) if i in ic_indices]
    for q, a in zip(ic_questions, ic_answers):
        prompt += prompt_template.format(q, a)
    prompt += prompt_template.format("{}", "").strip()
    return prompt


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def numpy_everything(seq):
    if isinstance(seq, torch.Tensor):
        return seq.detach().cpu().numpy()
    else:
        return [numpy_everything(subseq) for subseq in seq]


class TokenLengthException(Exception):
    pass
