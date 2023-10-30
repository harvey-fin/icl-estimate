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
    """ Prompt formatted with demonstrations, but not with test example
    """
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


class CBQATemplate:
    """ CBQA n shot default template """
    def __init__(self, dataset: DatasetDict, num_shots: int=4, template: str = "default"):
        self.ic_prompt: str = ""
        self.template = template
        self.train = dataset["train"]
        self.test = dataset["test"]
        self.prompts: List[str] = []
        self.labels: List[str] = self.test["answers"]

        # get in-context prompts
        ic_indices = random.sample(range(len(self.train["answers"])), num_shots)
        self.ic_examples = {}
        for key in self.train.keys():
            self.ic_examples[key] = [self.train[key][i] for i in ic_indices]
        for i, (q, a) in enumerate(zip(self.ic_examples["questions"], self.ic_examples["answers"])):
            self.ic_prompt = self._single_shot(self.ic_prompt, q, a)

        
        # form test prompts
        for q, a in zip(self.test["questions"], self.test["answers"]):
            self.prompts.append(self._single_shot(self.ic_prompt, q, a, include_answer=False))
    
    def _single_shot(self, prompt: str, q:str, a: str, include_answer: bool=True) -> str:
        prompt += q + "\nAnswer:"
        if include_answer:
            prompt += a + "\n\n"
        return prompt
        
    def get_prompts(self) -> List[str]:
        return self.prompts, self.labels


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
    NUM_SHOTS = 32 # Fixed for now
    SEED = 1 # Fixed for now, test multiple seeds later
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

def load(dataset_name, num_test, split=None):
    if dataset_name == "triviaqa":
        dataset = load_dataset("trivia_qa", "rc.nocontext")
        test_examples = dataset["validation"]["question"][:num_test]
        test_golds = [x['normalized_aliases'] for x in dataset["validation"]["answer"][:num_test]]
        train_examples = dataset["train"]["question"]
        train_golds = [x['normalized_value'] for x in dataset["train"]["answer"]]
        instruction = ""
        example_template = "Q: {}\nA: {}\n"

    elif dataset_name == "hotpotqa":
        if split is not None and split not in ["easy", "medium", "hard"]:
            raise ValueError
        dataset = load_dataset("hotpot_qa", "fullwiki")
        df_train = pd.DataFrame(dataset['train'])
        df_test = pd.DataFrame(dataset['validation'])
        # TODO: filter dataset if split is not None
        if split is None:
            train_examples = dataset['train']['question']
            train_golds = dataset['train']['answer']
            test_examples = dataset['validation']['question'][:num_test]
            test_golds = dataset['validation']['answer'][:num_test]
        else:
            train_examples = list(df_train[df_train['level']==split]["question"])
            train_golds = list(df_train[df_train['level']==split]["answer"])
            test_examples = list(df_test[df_test['level']==split]["question"])[:num_test] 
            test_golds = list(df_train[df_test['level']==split]["answer"])[:num_test]
        
        instruction = ""
        example_template = "Q: {}\nA: {}\n"

    elif dataset_name == "nq":
        dataset = load_dataset("nq_open")
        test_examples = dataset["validation"]["question"][:num_test]
        test_golds = dataset["validation"]["answer"][:num_test]
        train_examples = dataset["train"]["question"]
        train_golds = dataset["train"]["answer"]
        instruction = ""
        example_template = "Q: {}\nA: {}\n"

    elif dataset_name == "freebaseqa":
        dataset = load_dataset("freebase_qa")
        test_examples = dataset["validation"]["ProcessedQuestion"][:num_test]
        test_golds = [x["Answers"][0]['AnswersName'][0][0] for x in dataset["validation"]["Parses"][:num_test]]
        train_examples = dataset["train"]["ProcessedQuestion"]
        train_golds = [x["Answers"][0]['AnswersName'][0][0] for x in dataset["train"]["Parses"]]
        instruction = ""
        example_template = "Q: {}\nA: {}\n"

    elif dataset_name == "web_questions":
        dataset = load_dataset("web_questions")
        test_examples = dataset["test"]["question"][:num_test]
        test_golds = dataset["test"]["answers"][:num_test]
        train_examples = dataset["train"]["question"]
        train_golds = [x[0] for x in dataset["train"]["answers"]]
        instruction = ""
        example_template = "Q: {}\nA: {}\n"
    
    elif dataset_name == "squad":
        dataset = load_dataset("squad")
        test_examples = dataset['validation']['question'][:num_test]
        test_golds = [a['text'][0] for a in dataset['validation']['answers'][:num_test]]
        train_examples = dataset['train']['question']
        train_golds = [a['text'][0] for a in dataset['train']['answers']]
        instruction = ""
        example_template = "Q: {}\nA: {}\n"

    elif dataset_name == "adversarial_qa":
        dataset = load_dataset("adversarial_qa", "adversarialQA")
        test_examples = dataset['validation']['question'][:num_test]
        test_golds = [a["text"] for a in dataset['validation']['answers'][:num_test]]
        train_examples = dataset['train']['question']
        train_golds = [a['text'][0] for a in dataset['validation']['answers']]
        instruction = ""
        example_template = "Q: {}\nA: {}\n"
 
    elif dataset_name == 'jeopardy':
        dataset = load_dataset("jeopardy")
        # partitioned dataset
        p_dataset = dataset['train'][:100000]
        from sklearn.model_selection import train_test_split
        train_examples, test_examples, train_golds, test_golds = train_test_split(p_dataset['question'], p_dataset['answer'], test_size=0.05, random_state=42)
        test_examples = test_examples[:num_test]
        test_golds = test_golds[:num_test]
        instruction = ""
        example_template = "Q: {}\nA: {}\n"

    elif dataset_name == "numersense":
        dataset = load_dataset("numer_sense")
        test_examples = dataset['test_core']['sentence'][:num_test]
        test_golds = dataset['test_core']['target'][:num_test]
        train_examples = dataset['train']['sentence']
        train_golds = dataset['train']['target']
        
        instruction = ""
        example_template = "Q: {}\nA: {}\n"

    elif dataset_name == "search_qa":
        dataset = load_dataset("search_qa", "train_test_val")
        test_examples = dataset["test"]["question"][:num_test]
        test_golds = dataset["test"]["answer"][:num_test]
        train_examples = dataset["train"]["question"]
        train_golds = dataset["train"]["answer"]

        instruction = ""
        example_template = "Q: {}\nA: {}\n"

    elif dataset_name == "gigaword":
        dataset = load_dataset("gigaword")
        test_examples = dataset['test']['document'][:num_test]
        test_golds = dataset['test']['summary'][:num_test]
        train_examples = dataset["validation"]["document"]
        train_golds = dataset["validation"]["summary"]
        example_template = "Article: {}\nSummary: {}\n"
        instruction = "Summarize the article."
    
    elif dataset_name == 'amazon_reviews':
        dataset = load_dataset("amazon_us_reviews", "Wireless_v1_00")
        # partitioned dataset
        p_dataset = dataset['train'][:100000]
        from sklearn.model_selection import train_test_split
        train_examples, test_examples, train_golds, test_golds = train_test_split(p_dataset['review_body'], p_dataset['review_headline'], test_size=0.05, random_state=42) 
        test_examples = test_examples[:num_test]
        test_golds = test_golds[:num_test]

        example_template = "Article: {}\nSummary: {}\n"
        instruction = "Summarize the article."

    elif dataset_name == "de-en":
        dataset = load_dataset("wmt16", "de-en")
        train_examples = [d['de'] for d in dataset['validation']['translation']]
        train_golds = [d['en'] for d in dataset['validation']['translation']]
        test_examples = [d['de'] for d in dataset['test']['translation'][:num_test]]
        test_golds = [d['en'] for d in dataset['test']['translation'][:num_test]]
        
        example_template = "{}\n{}\n"
        instruction = "Translate from German to English"

    elif dataset_name == "en-de":
        dataset = load_dataset("wmt16", "de-en")
        train_examples = [d['en'] for d in dataset['validation']['translation']]
        train_golds = [d['de'] for d in dataset['validation']['translation']]
        test_examples = [d['en'] for d in dataset['test']['translation'][:num_test]]
        test_golds = [d['de'] for d in dataset['test']['translation'][:num_test]]
        
        example_template = "{}\n{}\n"
        instruction = "Translate from English to German"

    elif dataset_name == "ro-en":
        dataset = load_dataset("wmt16", "ro-en")
        train_examples = [d['ro'] for d in dataset['validation']['translation']]
        train_golds = [d['en'] for d in dataset['validation']['translation']]
        test_examples = [d['ro'] for d in dataset['test']['translation'][:num_test]]
        test_golds = [d['en'] for d in dataset['test']['translation'][:num_test]]
        
        example_template = "{}\n{}\n"
        instruction = "Translate from Romanian to English"

    elif dataset_name == "en-ro":
        dataset = load_dataset("wmt16", "ro-en")
        train_examples = [d['en'] for d in dataset['validation']['translation']]
        train_golds = [d['ro'] for d in dataset['validation']['translation']]
        test_examples = [d['en'] for d in dataset['test']['translation'][:num_test]]
        test_golds = [d['ro'] for d in dataset['test']['translation'][:num_test]]
        
        example_template = "{}\n{}\n"
        instruction = "Translate from English to Romanian"

    elif dataset_name == "fr-en":
        dataset = load_dataset("wmt14", "fr-en")
        train_examples = [d['fr'] for d in dataset['validation']['translation']]
        train_golds = [d['en'] for d in dataset['validation']['translation']]
        test_examples = [d['fr'] for d in dataset['test']['translation'][:num_test]]
        test_golds = [d['en'] for d in dataset['test']['translation'][:num_test]]
        
        example_template = "{}\n{}\n"
        instruction = "Translate from French to English"

    elif dataset_name == "en-fr":
        dataset = load_dataset("wmt14", "fr-en")
        train_examples = [d['en'] for d in dataset['validation']['translation']]
        train_golds = [d['fr'] for d in dataset['validation']['translation']]
        test_examples = [d['en'] for d in dataset['test']['translation'][:num_test]]
        test_golds = [d['fr'] for d in dataset['test']['translation'][:num_test]]
        
        example_template = "{}\n{}\n"
        instruction = "Translate from English to French"

    elif dataset_name == "cs-en":
        dataset = load_dataset("wmt16", "cs-en")
        train_examples = [d['cs'] for d in dataset['validation']['translation']]
        train_golds = [d['en'] for d in dataset['validation']['translation']]
        test_examples = [d['cs'] for d in dataset['test']['translation'][:num_test]]
        test_golds = [d['en'] for d in dataset['test']['translation'][:num_test]]
        
        example_template = "{}\n{}\n"
        instruction = "Translate from Czech to English"

    elif dataset_name == "en-cs":
        dataset = load_dataset("wmt16", "cs-en")
        train_examples = [d['en'] for d in dataset['validation']['translation']]
        train_golds = [d['cs'] for d in dataset['validation']['translation']]
        test_examples = [d['en'] for d in dataset['test']['translation'][:num_test]]
        test_golds = [d['cs'] for d in dataset['test']['translation'][:num_test]]
        
        example_template = "{}\n{}\n"
        instruction = "Translate from English to Czech"

    else:
        raise ValueError
    return train_examples, train_golds, test_examples, test_golds, example_template, instruction

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
