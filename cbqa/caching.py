from __future__ import annotations
import os
import random
import pickle
import torch
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
from torch.utils.data  import TensorDataset, DataLoader
from datasets import Dataset, DatasetDict
from typing import *
from load_cbqa import load_data
from utils import split
from config import ontology
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
    
class Task:
    """ Stores metadata about a task's type and source. """
    def __init__(
        self,
        name: str,
        task_type: str,
    ):
        self.task_type = task_type
        self.name = name
    
    def __str__(self) -> str:
        return f"Task({self.name})"

    def __eq__(self, other: Union[Task, str]) -> bool:
        if isinstance(other, Task):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            raise TypeError("Can only determine Task equality with another Task or name")

    def load(self) -> DatasetDict:
        data = load_data(self.name)
        return data

    def of_type(self, task_type: str) -> bool:
        return self.task_type == task_type
    

class TaskList:
    """ Abstracts a list of tasks, allowing easy filtering of tasks by type. """
    def __init__(self, tasks: dict[str, Task]):
        assert isinstance(tasks, dict)
        assert all(isinstance(task, Task) for task in tasks.values())
        self.tasks = tasks
        self.task_types = set(task.task_type for task in tasks.values())
    
    
    @staticmethod
    def load_cbqa() -> TaskList:
        """ Load Crossfit: CBQA tasks. defined in ontology["seq2seq"] """
        tasks = {}
        task_types = list(ontology.keys())
        task_types.remove("seq2seq")
        for task in ontology["cbqa"]:
            for t_type in task_types:
                if task in ontology[t_type]:
                    t = Task(name=task,
                            task_type=t_type)
            tasks[t.name] = t
        return TaskList(tasks)


    @property
    def names(self) -> List[str]:
        return list(self.tasks.keys())
    

    def split(self, n: int) -> List[TaskList]:
        """
        Returns List of n equally sized splits of the TaskList
        WARNING: Random!
        """
        tasks = self.tasks.items()
        shuffled_tasks = random.sample(tasks, len(tasks))
        split_tasks = list(split(shuffled_tasks, n))
        task_lists = [TaskList(dict(tasks)) for tasks in split_tasks]
        return task_lists


    def of_type(self, task_type: Union[str, List[str]]) -> TaskList:
        """ Filter tasks for a given type or list of types."""
        if isinstance(task_type, list):
            return self._of_types(task_type)
        assert task_type in self.task_types
        return TaskList({task_name: task for task_name, task in self.tasks.items() if task.task_type == task_type})

    
    def _of_types(self, task_types: List[str]) -> TaskList:
        """ Filters for tasks which fall into any given type."""
        return TaskList({task_name: task for task_name, task in self.tasks.items() if task.task_type == task_type})
    

    def of_ontology(self, ontology: Dict, task_type: List[str] = ontology.keys()) -> TaskList:
        """ Filter tasks for only those in the ontology """
        return self._of_names([task_name for t in task_type for task_name in ontology[t]])
    

    def __len__(self) -> int:
        return len(self.tasks)


    def __getitem__(self, key: Union[int, List[str], str]) -> Union[Task, TaskList]:
        """ Index by task name, integer index, or list of task names. """
        if type(key) == int:
            return list(self.tasks.values())[key]
        elif type(key) == list:
            return self._of_names(key)
        else:
            return self.tasks[key]


    def _of_names(self, task_names: List[str]) -> TaskList:
        return TaskList({task_name: self[task_name] for task_name in task_names})


    def __setitem__(self, key: str, value: Task):
        self.tasks[key] = value


    def __delitem__(self, key: str):
        del self.tasks[key]


    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks.values())


    def __str__(self) -> str:
        start = "["
        for task in self.tasks.values():
            start += task.name
            start += ", "
        start += "]"
        return start


    def __add__(self, other: TaskList) -> TaskList:
        assert len(set(self.tasks.keys()).intersection(other.tasks.keys())) == 0
        new_tasks = dict(chain.from_iterable([self.tasks.items(), other.tasks.items()]))
        return TaskList(new_tasks)


class InferenceResult:
    """ Caches the outputs of a single model on a task with a set of hyperparams."""
    def __init__(
        self, 
        confs: List[float], 
        embeds: List[np.ndarray],
        pca_embeds: List[np.ndarray],
        accs: List[float], 
        task: Task, 
        prompt: str, 
        seed: int, 
        num_shots: int, 
        model_name: str,
    ):
        assert isinstance(task, Task)
        assert isinstance(seed, int)
        self.confs = confs
        self.accs = accs
        self.pca_embeds = pca_embeds
        self.embeds = embeds
        self.task = task
        self.seed = seed
        self.prompt = prompt
        self.num_shots = num_shots
        self.model_name = model_name


    @classmethod
    def from_path(cls, path: str, pca_path:str, *args, **kwargs) -> InferenceResult:
        """ Cache an inference result from a .pkl file. """
        assert os.path.exists(path)
        with open(path, "rb") as f:
            d = pickle.load(f)
        confs = d["confs"]
        accs = d["accs"]
        embeds = d["embeds"]
        if pca_path:
            with open(pca_path, "rb") as f:
                d_pca = pickle.load(f)
            pca_embeds = d_pca
        else:
            pca_embeds=None
        return cls(accs=accs, confs=confs, embeds=embeds, pca_embeds=pca_embeds, *args, **kwargs)


    def is_same_source(self, other: Union[InferenceResult, Task]) -> bool:
        """
        This is used to remove all inferences from train set which:
        (a) are of the same Task, but different seed/prompt/etc.
        (b) are of a different Task but same HF source, e.g., 
            quartz-with_knowledge/quartz-no_knowledg
        (c) are of a different HF source but the same original dataset, e.g.,
            nq and kilt_nq
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
            return np.mean(self.pca_embeds[0], axis=0)
        else:
            return np.mean(self.pca_embeds, axis=0)


    def __len__(self) -> int:
        return len(self.confs)


    def __str__(self) -> str:
        return f"Inference({self.task}, seed={self.seed}, num_shots={self.num_shots})"


    def __getitem__(self, key: int) -> InferenceResult:
        """ Used to slice the number of examples in an inference to a fixed num. """
        inf_copy = InferenceResult(
            confs=self.confs[key], 
            accs=self.accs[key],
            embeds=self.embeds[key],
            pca_embeds=self.pca_embeds[key],
            task=self.task, 
            prompt=self.prompt, 
            seed=self.seed, 
            num_shots=self.num_shots, 
            model_name=self.model_name
        )
        return inf_copy


class InferenceCache:
    """ Caches a sweep of inferences from varying some params. """
    def __init__(self, inferences: List[InferenceResult]):
        assert isinstance(inferences, list)
        self.inferences = inferences


    @classmethod
    def load_from_path(cls, path: str, pca_path:str = None, task_list: TaskList = None) -> InferenceCache:
        """
        We infer the task_set from task_list, so for now require that
        all tasks in task_list are from the same source, e.g., crossfit
        """
        assert os.path.exists(path)
        inferences = []
        for fpath in tqdm(os.listdir(path)):
            fullpath = os.path.join(path, fpath)
            if pca_path:
                full_pca_path = os.path.join(pca_path, fpath)
                if not os.path.exists(full_pca_path):
                    continue
            else:
                full_pca_path=None

            task_name, prompt, num_shots, seed, model_name = \
                    cls._parse_filename(fpath)
            try:
                task = task_list[task_name]
            except KeyError as e:
                # Skip if it's not in task list
                continue
                #print(task_name, "not in task list")
            inf = InferenceResult.from_path(task=task, 
                    path=fullpath,
                    pca_path=full_pca_path,
                    seed=seed,
                    num_shots=num_shots,
                    model_name=model_name,
                    prompt=prompt,
            )
            inferences.append(inf)
        return InferenceCache(inferences)


    def _parse_filename(filename: str) -> Tuple:
        """
        Helper for load_from_path.
        """
        chunks = filename.split("_")    
        model_name = chunks[-1][:-4]
        seed = int(chunks[-2][4:])
        num_shots = int(chunks[-3][:-4])
        prompt = chunks[-4]

        if len(chunks) == 6:
            task_name = chunks[0] + "_" + chunks[1]
        elif len(chunks) == 7:
            task_name = chunks[0] + "_" + chunks[1] + "_" + chunks[2]
        elif len(chunks) == 8:
            task_name = chunks[0] + "_" + chunks[1] + "_" + chunks[2] + "_" + chunks[3]
        else:
            task_name = chunks[0]

        return task_name, prompt, num_shots, seed, model_name
    

    def limit(self, n: int) -> InferenceCache:
        """ Utility for setting all inferences to a fixed num examples n. """
        return InferenceCache([inf[:n] for inf in self.inferences])


    def get_embedloader(self, train:bool = True) -> Union[DataLoader, List[DataLoader]]:
        data = self.embeds[0]
        accs = self.y[0]
        if train:
            for idx in range(1, len(self.inferences)):
                x = self.embeds[idx]
                y = self.y[idx]
                data = np.concatenate((data, x), axis=0)
                accs = np.concatenate((accs, y), axis=0)
            x,y = torch.Tensor(x), torch.Tensor(y)
            dataset = TensorDataset(x, y)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            return dataloader
        else:
            data_list = []
            for idx in range(0, len(self.inferences)):
                x = self.embeds[idx]
                y = self.y[idx]
                x,y = torch.Tensor(x), torch.Tensor(y)
                dataset = TensorDataset(x, y)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
                data_list.append(dataloader)
            return data_list


    def exclude_conf(self, dim:int, batch_size: int=1) -> DataLoader:
        """ Makes a DataLoader of average embeddings binned by confidence profile """
        x = []
        y = []
        for idx in range(len(self.inferences)):
            conf = self.confs[idx]
            output_lengths = self.meta_data[idx]["output_lengths"]
            m = [v!=0 for v in output_lengths]
            check = [v for v in output_lengths if v!=0]
            if len(check)<20:
                continue
            locs = np.linspace(0, 100, num=dim)
            mask_conf = np.array(conf)[m]
            mask_y = np.array(self.y[idx])[m]
            values_conf = np.percentile(mask_conf, locs)
            x.append(values_conf)
            y.append(np.sum(mask_y)/len(conf))
        x, y = np.array(x), np.array(y)
        x, y = torch.Tensor(x), torch.Tensor(y)
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
        return dataloader


    def dataloader(self, dim: int, batch_size: int, metric:str="conf", method:str="mean") -> DataLoader:
        """ Makes a Dataloader from training meta-model by discretizing all confidence distros. """

        if metric=="conf":
            x = []
            y = []
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

            assert np.array(x).shape[1] == dim + 50, "incorrect dimension"


        # transform x into np array
        x, y = torch.Tensor(np.array(x)), torch.Tensor(np.array(y))
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def split(self, n: int) -> List[InferenceCache]:
        """
        Returns List of n equally sized splits of the cache
        WARNING: Random!
        """
        shuffled_cache = random.sample(self.inferences, len(self.inferences))
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
    def embeds(self):
        return [inf.embeds for inf in self.inferences]

    @property
    def pca_embeds(self):
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

    @property
    def mean_confs(self) -> List[float]:
        return [inf.mean_conf for inf in self.inferences]

    def of_size(self, model_size:str) -> InfereceCache:
        """ Filter only inferences of a given model size """
        return InferenceCache([inf for inf in self.inferences if inf.model_name == model_size])


    def of_task(self, task: Union[Task, str, TaskList, List[str]]) -> InferenceCache:
        """ Filter only inferences of a given task(s). """
        if isinstance(task, list) or isinstance(task, TaskList):
            return self._of_any_task(task)
        assert isinstance(task, str) or isinstance(task, Task)
        return InferenceCache([inf for inf in self.inferences if inf.task == task])
    
    def _of_any_task(self, tasks: Union[TaskList, List[str]]) -> InferenceCache:
        """ Helper to get inferences from any of a set of tasks. """
        return InferenceCache([inf for inf in self.inferences if inf.task in tasks])

    def top_seeds(self, n:int) -> InferenceCache:
        """return the first n seeds of a inference"""
        seed_list = list(range(1, n+1))
        return self.of_seeds(seed_list) 
    
    def of_seeds(self, seed_list: List[int]) -> InferenceCache:
        """Filter only inferences of a given list of seeds"""
        return InferenceCache([inf for inf in self.inferences if inf.seed in seed_list])

    def of_shots(self, num_shots: int) -> InferenceCache:
        """ Filter only inferences of a given number of shots. """
        return InferenceCache([inf for inf in self.inferences if inf.num_shots == num_shots])

    def of_type(self, task_type: str) -> InferenceCache:
        """ Filter only inferences of a given task type. """
        return InferenceCache([inf for inf in self.inferences if inf.task.task_type == task_type])
    
    def exclude_related(self, other: Union[InferenceResult, Task, List[InferenceResult], InferenceCache]) -> InferenceCache:
        """
        Return a new cache with all _similar_ inferences to other gone.
        See InferenceResult.is_same_source for meaning of _similar_
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

    @property
    def tasks(self) -> TaskList:
        return TaskList({inf.task.name: inf.task for inf in self.inferences})


    def __len__(self) -> int:
        return len(self.inferences)

    def __getitem__(self, key: int) -> InferenceResult:
        return self.inferences[key]
    
    def __setitem__(self, key: int, value: InferenceResult):
        self.inferences[key] = value
    
    def __delitem__(self, key: int):
        del self.inferences[key]

    def __iter__(self) -> Iterator[InferenceResult]:
        return iter(self.inferences)

    def __add__(self, other: InferenceCache) -> InferenceCache:
        assert isinstance(other, InferenceCache)
        return InferenceCache(self.inferences + other.inferences)

    def __str__(self) -> str:
        start = "["
        for inf in self.inferences:
            #start += inf.task
            start += str(inf) 
            start += ", "
        start += "]"
        return start 
