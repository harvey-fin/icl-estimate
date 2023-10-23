import numpy as np
from collections import Counter
from sklearn import metrics
from utils import normalize
from typing import *

class LLaMAScorer:
    def __init__(self, answer_checker: str, confidence_metric: str):
        if answer_checker == "exact_match":
            self.checker = self._checker_em
        elif answer_checker == "f1":
            self.checker = self._checker_f1
        if confidence_metric == "label_normalize":
            self.conf_metric = self._label_normalize

    def score(self, probs: List[np.ndarray], labels: List[int]) -> Union[List[str], List[float]]:
        """ Function for scoring the confidence and accuracy. Label is the corresponding integer for choices """
        # normalize across all probability
        outputs = [np.argmax(p) for p in probs]
        accs = [int(self.checker(o, l)) for o, l in zip(outputs, labels)]

        return accs, probs

    def _checker_em(self, prediction, label) -> bool:
        return prediction == label

    
    def _label_normalize(self, probs: List[np.ndarray]) -> List[float]:
        return [np.max(p)/np.sum(p) for p in probs]


    def _checker_f1(self, prediction, label):
        prediction_tokens = prediction.split()
        ground_truth_tokens = lable.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
