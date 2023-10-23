from __future__ import annotations
import os
import random
import torch
import numpy as np
import pickle
from typing import *
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from config import ontology
from load_crossfit import load_data
from torch.utils.data import TensorDataset, DataLoader
from utils import split

class Task:
    """ Stores metadata about subsets of the MMLU and MCQA benchmark """
    def __init__(
        self,
        subset: str,
        category: str,
    ):
        self.subset = subset
        self.category = category


    def __str__(self) -> str:
        if "crossfit" in self.subset:
            return self.subset
        else:
            return f"MMLU_{self.subset}"


    def __eq__(self, other: Union[Task, str]) -> bool:
        if isinstance(other, Task):
            return self.subset == other.subset
        elif isinstance(other, str):
            return self.subset == other
        else:
            raise TypeError("Can only determine equality with another MMLU subset object or name")


    def load(self) -> DatasetDict:
        if "crossfit" in self.subset:
            data = load_data(self.subset)
        else:
            data = load_dataset("cais/mmlu", self.subset)
        return data


    def of_category(self, category: str) -> bool:
        return self.category == category


class TaskList:
    """ 
    A list of tasks from the MMLU datset, allowing easy filtering of tasks by type. 
    tasks are stored as a dictionary with subset as the key, Task as the value
    """
    def __init__(self, tasks: dict[str, Task]):
        assert isinstance(tasks, dict)
        assert all(isinstance(task, Task) for task in tasks.values())
        self.tasks = tasks
        self.categories = list(ontology.keys())
        self.categories.remove("mmlu")


    @staticmethod
    def load_mmlu() -> TaskList:
        """ Load MMLU tasks with their supercategory """
        tasks = {}
        categories = list(ontology.keys())
        categories.remove("mmlu")
        for task in ontology["mmlu"]:
            for c in categories:
                if task in ontology[c]:
                    t = Task(subset=task,
                            category = c)
            tasks[t.subset] = t
        return TaskList(tasks)


    @staticmethod
    def load_crossfit() -> TaskList:
        """ Load Crossfit: MCQA tasks. Their supercategory would be crossfit """
        tasks = {}
        for task in ontology["crossfit"]:
            t = Task(subset=task, category="crossfit")
            tasks[t.subset] = t
        return TaskList(tasks)
            

    @property
    def names(self) -> List[str]:
        return list(self.tasks.keys())


    def split(self, n: int) -> List[TaskList]:
        """ Returns a list of n equally sized splits fo the TaskList """
        tasks = self.tasks.items()
        shuffled_tasks = random.sample(tasks, len(tasks))
        split_tasks = list(split(shuffled_tasks, n))
        task_lists = [TaskList(dict(tasks)) for tasks in split_tasks]
        return task_lists


    def _of_category(self, category: str) -> TaskList:
        return TaskList({task_name: task for task_name, task in self.tasks.items() if task.of_category(category)})


    def _of_categories(self, categories: List[str]) -> TaskList:
        return TaskList({task_name: task for task_name, task in self.tasks.items() if task.category in categories})


    def _of_names(self, task_names: List[str]) -> TaskList:
        return TaskList({task_name: task for task_name, task in self.tasks.items() if task_name in task_names})


    def __len__(self) -> int:
        return len(self.tasks)


    def __getitem__(self, key: Union[int, List[str], str]) -> Union[Task, TaskList]:
        """ Index by task category, subset, integer index, or list of categories and subsets. """
        if type(key) == int:
            return list(self.tasks.values())[key]
        elif type(key) == str:
            if key in self.categories:
                return self._of_category(key)
            else:
                return self.tasks[key]
        else:
            return self.tasks[key]


    def __setitem__(self, key: str, value: Task):
        self.tasks[key] = value


    def __delitem__(self, key: str):
        del self.tasks[key]


    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks.values())


    def __str__(self) -> str:
        return ' '.join(self.tasks.keys())


    def __add__(self, other: TaskList) -> TaskList:
        assert len(set(self.tasks.keys()).intersection(other.tasks.keys())) == 0
        new_tasks = dict(chain.from_iterable([self.tasks.items(), other.tasks.items()]))
        return TaskList(new_tasks)


class InferenceResult:
    """ Caches the outputs of a single model on a task with a set of hyperparameters """
    def __init__(
        self,
        probs: List[np.ndarray],
        accs: List[int],
        embeds: List[np.ndarray],
        pca_embeds: List[np.ndarray],
        task: Task,
        prompt: str,
        seed: int,
        num_shots: int,
        model_name: str,
    ):
        assert isinstance(task, Task)
        assert isinstance(seed, int)

        self.probs = probs
        self.confs = [np.max(p)/np.sum(p) for p in self.probs]
        self.logits = np.log(self.probs)
        self.accs = accs     # list of accuracy
        self.embeds = embeds
        self.pca_embeds = pca_embeds,
        self.task = task
        self.seed = seed
        self.prompt = prompt
        self.num_shots = num_shots
        self.model_name = model_name


    @classmethod
    def from_path(cls, path: str, pca_path:str, *args, **kwargs) -> InferenceResult:
        """Cache an inference result from a pickle file"""
        assert os.path.exists(path)
        with open(path, "rb") as f:
            d = pickle.load(f)
        probs = d["confs"]
        accs = d["accs"]
        embeds = d["embeds"]
        if pca_path:
            with open(pca_path, "rb") as f:
                d_pca = pickle.load(f)
            pca_embeds = d_pca
        else:
            pca_embeds = None
        return cls(accs=accs, probs=probs, embeds=embeds, pca_embeds=pca_embeds, *args, **kwargs)


    def is_same_source(self, other: Union[InferenceResult, Task]) -> bool:
        """
        function used for removing inferences from the training set (for CV):
        are of the same task, but different seeds/prompt
        """
        if isinstance(other, InferenceResult):
            other = other.task
        return other == self.task


    @property
    def mean_score(self) -> float:
        return np.mean(self.accs)

    @property
    def mean_conf(self) -> float:
        return np.mean(self.confs)

    @property
    def mean_embed(self) -> np.ndarray:
        return np.mean(self.embeds, axis=0)

    @property
    def mean_pca_embed(self) -> np.ndarray:
        if isinstance(self.pca_embeds[0], tuple):
            return np.mean(self.pca_embeds[0][0], axis=0)
        else:
            return np.mean(self.pca_embeds[0][0], axis=0)


    def scaled_conf(self, temp:float) -> np.ndarray:
        temp_logits = self.logits/temp
        return np.max(np.exp(temp_logits), axis=1)/np.sum(np.exp(temp_logits), axis=1)


    def mean_scaled_conf(self, temp:float) -> np.ndarray:
        return np.mean(self.scaled_conf(temp), axis=0)


    def __len__(self) -> int:
        return len(self.confs)


    def __str__(self) -> str:
        return f"Inference Result on {self.task}, seed = {self.seed}, num_shots = {self.num_shots}"


    def __getitem__(self, key:int) -> InferenceResult:
        """
        Used to slice the number of examples in inference
        """
        inf_copy = InferenceResult(
            probs = self.probs[key],
            accs = self.accs[key],
            embeds = self.embeds[key],
            pca_embeds = self.pca_embeds[key],
            task = self.task,
            prompt = self.prompt,
            seed = self.seed,
            num_shots = self.num_shots,
            model_name = self.model_name,
            )
        return inf_copy


    def is_same_source(self, other: Union[InferenceResult, Task]) -> bool:
        """
        Used to remove all inferences from train set:
        (a) are of same Task, but different seed/shot
        """
        if isinstance(other, InferenceResult):
            other = other.task
        return other == self.task


class InferenceCache:
    """
    Caches a sweep of infernces with varying some parameters
    structure: Inference Cache would be a list of InfereceResult
    """
    def __init__(self, inferences: List[InferenceResult]):
        assert isinstance(inferences, list)
        self.inferences = inferences


    @classmethod
    def load_from_path(cls, path: str, pca_path:str = None, task_list: TaskList = None) -> InferenceCache:
        """
        Load a list of InferenceResult from pickle files as InferenceCache
        """
        assert os.path.exists(path)
        inferences = []

        # load every file in the path directory
        for fpath in tqdm(os.listdir(path)):
            fullpath = os.path.join(path, fpath)
            
            if pca_path:
                full_pca_path = os.path.join(pca_path, fpath)
                if not os.path.exists(full_pca_path):
                    full_pca_path=None
            else:
                full_pca_path=None

            task_name, prompt, num_shots, seed, model_name = \
                    cls._parse_filename(fpath)
            try:
                task = task_list[task_name]
            except KeyError as e:
                continue
            
            # load file as InferenceResult by calling from_path
            inf = InferenceResult.from_path(task=task,
                    path=fullpath,
                    pca_path=full_pca_path,
                    prompt=prompt,
                    seed=seed,
                    num_shots=num_shots,
                    model_name=model_name)
            inferences.append(inf)

        return InferenceCache(inferences)


    def _parse_filename(filename: str) -> Tuple:
        """
        parse the filename for the pickle file for loading data
        example file_name: {subset}_default_5shot_seed0_llama-13b.pkl
        """
        chunks = filename.split("_")
        model_name = chunks[-1][:-4]
        seed = int(chunks[-2][4:])
        num_shots = int(chunks[-3][:-4])
        prompt = chunks[-4]
        if len(chunks)==6:
            task_name = chunks[0] + "_" + chunks[1]
        elif len(chunks) == 7:
            task_name = chunks[0] + "_" + chunks[1] + "_" + chunks[2]
        else:
            task_name = chunks[0]
        
        return task_name, prompt, num_shots, seed, model_name


    def limit(self, n: int) -> InferenceCache:
        """
        set all inferences to a fixed number of examples n
        """
        return InferenceCache([inf[:n] for inf in self.inferences])


    def dataloader(self, batch_size:int, dim:int, metric:str="conf", method:str="mean") -> DataLoader:
        """ Makes a dataloader from training meta-model by discretizing all confidence distros """

        if metric=="conf":
            x = []
            y = []
            # load percentage confidence vector for each inference
            for idx in range(len(self.inferences)):
                locs = np.linspace(0, 100, num=int(dim))
                try:
                    values_conf = np.percentile(self.confs[idx], locs)
                except IndexError as e:
                    continue
                x.append(values_conf)
                y.append(np.mean(self.accs[idx]))
        elif metric == "embed":
            if method == "mean":
                x = self.mean_embeds
                y = self.mean_scores
            else:
                x = self.embeds
                y = self.accs
        elif metric == "pca_embed":
            if method == "mean":
                x = self.mean_pca_embeds
                y = self.mean_scores
            else:
                x = self.pca_embeds
                y = self.accs
        elif metric == "conf_embed":
            x = []
            y = []
            for idx in range(len(self.inferences)):
                locs = np.linspace(0, 100, num=int(dim))

                try:
                    values_conf = np.percentile(self.confs[idx], locs)
                except IndexError as e:
                    continue
                
                x.append(np.concatenate((values_conf, np.array(self.mean_pca_embeds[idx]))))
                y.append(np.mean(self.accs[idx]))
        
            assert np.array(x).shape[1] == dim + 50, "incorrect dimmension"

        # transform x into np array
        x, y = torch.Tensor(np.array(x)), torch.Tensor(np.array(y))
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader


    def split(self, n: int) -> List[InferenceCache]:
        """
        Returns a list of n equally sized splits of the cache
        """
        shuffled_cache = random.sample(self.inferences, len(self.inference))
        split_inferences = list(split(shuffled_cache, n))
        caches = [InferenceCache(infs) for infs in split_inferences]
        return caches

    @property
    def confs(self) -> List[List[float]]:
        return [inf.confs for inf in self.inferences]

    @property
    def accs(self) -> List[List[int]]:
        return [inf.accs for inf in self.inferences]

    @property
    def embeds(self) -> List[List[ndarray]]:
        return [inf.embeds for inf in self.inferences]

    @property
    def pca_embeds(self) -> List[List[ndarray]]:
        return [inf.pca_embeds for inf in self.inferences]

    @property
    def mean_embeds(self) -> List[np.ndarray]:
        return [inf.mean_embed for inf in self.inferences]

    @property
    def mean_pca_embeds(self) -> List[np.ndarray]:
        return [inf.mean_pca_embed for inf in self.inferences]

    @property
    def mean_scores(self) -> List[float]:
        return [inf.mean_score for inf in self.inferences]

    def scaled_confs(self, temp: float) -> List[List[float]]:
        return [inf.scaled_conf(temp) for inf in self.inferences] 
    
    def mean_scaled_confs(self, temp: float) -> List[float]:
        return [inf.mean_scaled_conf(temp) for inf in self.inferences] 

    @property
    def mean_confs(self) -> List[float]:
        return [inf.mean_conf for inf in self.inferences]

    @property
    def tasks(self) -> TaskList:
        return TaskList({inf.task.subset: inf.task for inf in self.inferences})

    def of_size(self, model_size: str) -> InferenceCache:
        """ Filter only inferences of a given model size """
        return InferenceCache([inf for inf in self.inferences if inf.model_name == model_size])

    def of_task(self, task: Union[Task, str, TaskList, List[str]]) -> InferenceCache:
        """ Filter only inferences of a given task(s). """
        if isinstance(task, list) or isinstance(task, TaskList):
            return self._of_any_task(task)
        assert isinstance(task, str) or isinstance(task, Task)
        return InferenceCache([inf for inf in self.inferences if inf.task==task])

    def _of_any_task(self, tasks: Union[TaskList, List[str]]) -> InferenceCache:
        """ Helper to get inferences from any of a set of tasks. """
        return InferenceCache([inf for inf in self.inferences if inf.task in tasks])

    def top_seeds(self, n: int) -> InferenceCache:
        """ return the first n seeds of an inference """
        seed_list = list(range(1, n+1))
        return self.of_seeds(seed_list)

    def of_seeds(self, seed_list: List[int]) -> InferenceCache:
        """ Filter only inferences of a given list of seeds """
        return InferenceCache([inf for inf in self.inferences if inf.seed in seed_list])

    def of_shots(self, num_shots: int) -> InferenceCache:
        """ Filter only inferences of a given number of shots """
        return InferenceCache([inf for inf in self.inferences if inf.num_shots == num_shots])
    
    def of_prompt(self, prompt: str) -> InferenceCache:
        """ Filter only inferences of a given prompt template """
        return InferenceCache([inf for inf in self.inferences if inf.prompt == prompt])

    def __iter__(self) -> Iterator[InferenceResult]:
        return iter(self.inferences)

    def __add__(self, other: InferenceCache) -> InferenceCache:
        assert isinstance(other, InferenceCache)
        return InferenceCache(self.inferences + other.inferences)

    def __delitem__(self, key: int):
        del self.inferences[key]

    def __len__(self) -> int:
        return len(self.inferences)

    def __getitem__(self, key: int) -> InferenceResult:
        return self.inferences[key]

    def __setitem__(self, key: int, value: InferenceResult):
        self.inferences[key] = value

    def exclude_related(self, other: Union[InferenceResult, Task, List[InferenceResult], InferenceCache]) -> InferenceCache:
        """
        Return a new cache with all similar inferences to other gone.
        See InferenceResult.is_same_source for meaning of similar
        """
        if isinstance(other, list) or isinstance(other, InferenceCache):
            return self._exclude_all_related(other)
        elif isinstance(other, InferenceResult) or isinstance(other, Task):
            return InferenceCache([inf for inf in self.inferences if not inf.is_same_source(other)])
        else:
            raise TypeError

    def _exclude_all_related(self, other: Union[List[InferenceResult], InferenceCache]) -> InferenceCache:
        assert isinstance(other, InferenceCache) or isinstance(other, list)
        assert all(isinstance(inf, InferenceResult) for inf in other)
        cache = self
        for inf in other:
            cache = cache.exclude_related(inf)
        return cache

