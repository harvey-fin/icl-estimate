import numpy as np
from sklearn import metrics
from utils import normalize
from collections import Counter
from matplotlib import pyplot as plt
import scipy

class Seq2SeqScorer:
    def __init__(self, score_metric, answer_checker, do_normalize=True, do_plot=False, plot_dir=None):
        self.do_normalize = do_normalize
        self.do_plot = do_plot
        self.plot_dir = plot_dir
        if score_metric == "auac":
            self.metric = self._metric_auac
        if score_metric == "auroc":
            self.metric = self._metric_auroc
        if score_metric == "spearman":
            self.metric = self._metric_spearman
        elif score_metric is None:
            self.metric = self._metric_none
        else:
            raise ValueError
        if answer_checker == "exact_match":
            self.checker = self._checker_em
        elif answer_checker == "f1":
            self.checker = self._checker_f1
        elif answer_checker == "bleu":
            self.checker = self._checker_bleu
        else:
            raise ValueError

    def score(self, predictions, confidences, labels):
        assert isinstance(confidences, np.ndarray)
        assert isinstance(predictions, list)
        assert isinstance(labels, list)
        assert len(predictions) == len(labels)
        assert len(predictions) == len(confidences)
        if self.do_normalize:
            predictions = list(map(normalize, predictions))
            if isinstance(labels[0], list):
                labels = [list(map(normalize, aliases)) for aliases in labels]
            else:
                labels = list(map(normalize, labels))
        if isinstance(labels[0], list):
            reference_scores = np.array([max([self.checker(pred, alias) for alias in aliases])
                    for pred, aliases
                    in zip(predictions, labels)])
        else:
            reference_scores = np.array([self.checker(pred, label) 
                    for pred, label 
                    in zip(predictions, labels)])
        self.reference_scores = reference_scores
        if self.do_plot:
            assert self.plot_dir is not None, "Need to pass in path to save plot!"
            self.plot(confidences, reference_scores, save_dir=self.plot_dir)
        assert isinstance(reference_scores, np.ndarray)
        #print(reference_scores)
        #print(confidences)
        metric_result = self.metric(reference_scores, confidences)
        results = {"score1": reference_scores}
        if metric_result is not None:
            results["score2"] = metric_result.correlation
            results["score2_pval"] = metric_result.pvalue
        return results

    def plot(self, predictions, ground_truths, save_dir):
        plt.figure()
        plt.tight_layout()
        plt.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        coef = np.polyfit(predictions,ground_truths,1)
        poly1d_fn = np.poly1d(coef) 
        plt.scatter(predictions, ground_truths, alpha=0.5, color='cyan')
        plt.ylim([0, 1])
        plt.plot(predictions, poly1d_fn(predictions), '--k')
        plt.xlabel("Confidence")
        plt.ylabel("Performance")
        plt.savefig(save_dir)

    def _checker_em(self, prediction, label):
        return prediction == label

    def _checker_f1(self, prediction, label):
        # from https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
        prediction_tokens = prediction.split()
        ground_truth_tokens = label.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


    def _checker_bleu(self, prediction, ground_truth):
        raise NotImplementedError

    def _metric_auroc(self, references, confidences):
        if np.sum(references) == 0:
            return 0
        return metrics.roc_auc_score(references, confidences)

    def _metric_spearman(self, references, confidences):
        return scipy.stats.spearmanr(references, confidences)
    
    def _metric_auac(self, references, confidences):
        assert isinstance(confidences, np.ndarray)
        assert isinstance(references, np.ndarray)
        _, references = list(
            zip(*sorted(zip(references, confidences), reverse=True, key=lambda x: x[0]))
        )
        x = np.arange(1, len(references) + 1)
        cumulative_scores = np.cumsum(references) / x
        return metrics.auc(x / len(x), cumulative_scores)
    
    def _metric_none(self, references, confidences):
        return None
