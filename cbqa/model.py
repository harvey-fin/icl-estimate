import numpy as np
from torch.nn import functional as F
import torch
from torch import nn, optim
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, OPTForCausalLM
from scipy.optimize import curve_fit
from utils import batch, numpy_everything, NoPrint, TokenLengthException
from transformers import logging
from matplotlib import pyplot as plt
from simcse import SimCSE
from typing import *
logging.set_verbosity_error()


class OPTForSeq2Seq:
    def __init__(self, model_name, max_length=2048, device="cuda", batch_size=None, length_penalty=None, max_new_tokens=None, early_stopping=None, *args, **kwargs):
        self.max_length = max_length
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        # Use half-precision for opt, else full
        if model_name.startswith("facebook/opt"):
            self.precision = torch.float16
        else:
            self.precision = torch.float32
        print("Setting up model...")
        self.model = OPTForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=self.precision,
            max_length=self.max_length,
            *args,
            **kwargs).to(self.device)
        print("done!")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.tokenizer.padding_side = "left"
        self.newline_token = self.tokenizer.encode("\n")[1]

    def _truncate_for_beam(self, outputs, probs):
        outputs_new, probs_new = [], []
        for op, pb in zip(outputs, probs):
            op1, pb1 = list(zip(*[self._truncate_output(o,p) for o,p in zip(op, pb)]))
            outputs_new.append(op1)
            probs_new.append(pb1)
        return outputs_new, probs_new
            
    def __call__(self, examples, embedding_metric="example_mean", confidence_metric=None, do_sample=False, num_beams=None, threshold=1.0):
        if "beam" in confidence_metric:
            self.model.config.eos_token_id=50118
            #print(examples)
            outputs, probs, seq_scores = self.forward_beam(examples, num_beams=num_beams)
            #outputs, probs = self.forward_beam(examples, num_beams=num_beams)
            # softmax over dim-1 for each example, beam
            #probs = [[F.softmax(p.float(),dim=-1).numpy() for p in probs_for_example] for probs_for_example in probs]
            # have to put this in a separate function, annoyingly
            outputs, probs = self._truncate_for_beam(outputs, probs)
            output_sequences = [self.tokenizer.batch_decode(
                beam_outputs, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True) for beam_outputs in outputs]
            #print(output_sequences)
            output_sequences = [[s.strip() for s in sq] for sq in output_sequences]
        elif "greedy" in confidence_metric:
            self.model.config.eos_token_id=50118
            outputs, probs, embeddings = self.forward_single(examples)
            # softmax over dim-1 for each example
            #probs = [F.softmax(p.float(),dim=-1).numpy() for p in probs]
            outputs, probs = list(zip(*[self._truncate_output(o,p) for o,p in zip(outputs, probs)]))
            output_sequences = self.tokenizer.batch_decode(
                outputs, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True)
            #print(output_sequences)
            output_sequences = [s.strip() for s in output_sequences]
        elif "random" in confidence_metric:
            raise NotImplementedError
        else:
            raise NotImplementedError
        outputs = numpy_everything(outputs)
        if confidence_metric is None:
            return output_sequences, outputs, probs
        elif confidence_metric == "random":
            confidences = self._confidence_random(probs)
        elif confidence_metric == "greedy_perplexity":
            confidences = self._confidence_perplexity(outputs, probs)
        elif confidence_metric == "greedy_loglikelihood":
            confidences = self._confidence_ll(outputs, probs, debug='greedy')
        elif confidence_metric == "beam_perplexity":
            confidences = self._confidence_perplexity([x[0] for x in outputs], [x[0] for x in probs])
            output_sequences = [os[0] for os in output_sequences]
        elif confidence_metric == "beam_loglikelihood":
            confidences = self._confidence_ll([x[0] for x in outputs], [x[0] for x in probs], debug='beam')
            output_sequences = [os[0] for os in output_sequences]
        elif confidence_metric == "beam_aggregation":
            #confidences = self._confidence_beam(outputs, output_sequences, probs)
            confidences = self._confidence_beam_ss(seq_scores, output_sequences, threshold=threshold)
            # get only top beam of each
            output_sequences = [os[0] for os in output_sequences]
        elif confidence_metric == "beam_ss":
            confidences = np.array([s[0].item() for s in seq_scores])
            output_sequences = [os[0] for os in output_sequences]
        elif confidence_metric == "random_perplexity":
            confidences = self._confidence_perplexity([x[0] for x in outputs], [x[0] for x in probs])
            output_sequences = [os[0] for os in output_sequences]
        elif confidence_metric == "random_loglikelihood":
            confidences = self._confidence_ll([x[0] for x in outputs], [x[0] for x in probs])
            output_sequences = [os[0] for os in output_sequences]
        elif confidence_metric == "random_aggregation":
            confidences = self._confidence_beam([x[0] for x in outputs], [x[0] for x in probs], threshold=threshold)
            output_sequences = [os[0] for os in output_sequences]
        else:
            raise ValueError("Confidence metric " + confidence_metric + " not in known metrics")
        
        if embedding_metric == "example_mean":
            # return a numpy array for the embedding
            embeddings = np.array([embed.numpy() for embed in embeddings]).astype(np.float32)
            #embeddings = np.array([torch.mean(embed.reshape(-1)).numpy().item() for embed in embeddings]).astype(np.float32)
    
        return output_sequences, confidences, embeddings

    def _truncate_output(self, model_output, probs):
        model_output_tokens = model_output.tolist()
        if model_output_tokens[0] == self.newline_token:
            if len(model_output_tokens) == 1:
                return model_output, probs
            model_output = model_output[1:]
            probs = probs[1:]
            return self._truncate_output(model_output, probs)
        if not self.newline_token in model_output_tokens:
            return model_output, probs
        else:
            idx = model_output_tokens.index(self.newline_token)
            return model_output[:idx], probs[:idx]
    
    def forward(self, examples, mode="single"):
        if mode == "single":
            return forward_single(examples)
        elif mode == "batch":
            return forward_batch(examples)
        else:
            raise ValueError
    
    def embed(self, examples: List[str]) -> List[torch.Tensor]:
        """ Return the last-token, last-layer embedding from the model of examples
        """

        embeddings = []
        for example in tqdm(examples):
            inputs = self.tokenizer(
                example,
                max_length = self.max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)
            res = self.model.generate(
                inputs,
                output_hidden_states=True,
                return_dict_in_generate = True,
                max_new_tokens=self.max_new_tokens,
                length_penalty=self.length_penalty,
                #begin_suppress_tokens=[self.model.config.eos_token_id],
            )
            hidden_states = res.hidden_states[0][-1][:,-1,:].detach().cpu()
            for hs_per_example in hidden_states:
                embeddings.append(hs_per_example.numpy().reshape(-1))
        return np.array(embeddings, dtype=object).astype(np.float32)
    

    def forward_batch(self, examples):
        """
        input: Batch of examples, formatted with prompt included.
        output: List[pred], List[probs]
           pred are Tensor(seqlen)
            probs are Tensor(seqlen, vocab_size)
        """
        outputs = []
        probs = []
        for example_batch in batch(examples, n=self.batch_size):
            inputs = self.tokenizer(
                example_batch,
                return_tensors="pt",
                padding=True, 
                max_length=self.max_length, 
                truncation=True,
                add_special_tokens=False)
            inputs = {key:value.to(device) for key, value in inputs.items()}
            res = self.model.generate(
                **inputs,
                output_scores=True,
                return_dict_in_generate=True,
            )
            output_batch = res.sequences[:, inputs.shape[1]:]
            outputs.extend(output_batch)
            unsqueeze_scores = [score.unsqueeze(1) for score in res.scores]
            probs.extend(torch.cat(unsqueeze_scores, axis=1).cpu())
        return outputs, probs

    def forward_single(self, examples: List[str]) -> List[torch.Tensor]:
        """
        input: Batch of examples, formatted with prompt included.
        output: List[pred], List[probs]
            outputs: list of output tokens
            probs: list of prob tensors 
            embeddings: list of tensors
        """
        outputs = []
        probs = []
        embeddings = []
        for example in tqdm(examples):
            inputs = self.tokenizer(
                example,
                return_tensors="pt"
            ).input_ids.to(self.device)
            if inputs.shape[1] > self.model.config.max_length:
                raise TokenLengthException
            res = self.model.generate(
                inputs,
                output_scores=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                max_new_tokens=self.max_new_tokens,
                length_penalty=self.length_penalty,
                #begin_suppress_tokens=[self.model.config.eos_token_id],
            )
            hidden_states = res.hidden_states[0][-1][:,-1,:].detach().cpu()
            for hs_per_example in hidden_states:
                embeddings.append(hs_per_example)
            output_token = res.sequences[0][-len(res.scores):]
            outputs.append(output_token)
            # score (1, 50272)
            # cat = (seqlen, 50272)
            # Do a gather here?
            probs.append(torch.FloatTensor([F.softmax(x, dim=-1).max() for x in res.scores]))
            #probs.append(torch.cat(res.scores, axis=0).cpu())
        return outputs, probs, embeddings
    
    def forward_beam(self, examples, num_beams=None):
        """
        unbatched
        input: Batch of examples, formatted with prompt included.
        output: preds (bsize x num_beams x seqlen), probs (bsize x num_beams x seqlen)
            pred: list of output tokens
            probs: list of prob tensors 
        """
        outputs = []
        # (num_examples, num_beams, num_tokens)
        probs = []
        seq_scores = []
        for example in tqdm(examples):
            inputs = self.tokenizer(
                example,
                return_tensors="pt"
            ).input_ids.to(self.device)

            if inputs.shape[1] > self.model.config.max_length:
                raise TokenLengthException

            res = self.model.generate(
                inputs,
                output_scores=True,
                return_dict_in_generate=True,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_new_tokens=self.max_new_tokens,
                length_penalty=self.length_penalty,
                begin_suppress_tokens=[self.model.config.eos_token_id],
            ) # (num_beams, max_num_tokens)
            #print(res.beam_indices[0][:10])
            ll_scores = self.model.compute_transition_beam_scores(res.sequences, res.scores, res.beam_indices, self.tokenizer.eos_token_id)
            ll_scores[:, 0] = torch.max(F.softmax(res.scores[0], dim=-1), dim=-1).values.log()
            probs.append(torch.exp(ll_scores).cpu())
            sequence_outputs = res.sequences[:,-len(res.scores):]
            outputs.append(sequence_outputs)
            seq_scores.append(res.sequences_scores)
            #cat_scores = torch.cat([s.unsqueeze(0) for s in res.scores], axis=0)
            #cat_scores = cat_scores.transpose(0, 1)
            #probs.append(cat_scores.cpu())
        return outputs, probs, seq_scores

    def _confidence_random(self, probs):
        """
        input: probs (bsize, seqlen, vocab_size)
        output: confidences (bsize) np.array
        """
        batch_size = len(probs)
        return np.random.rand(batch_size)

    def _confidence_perplexity(self, pred, probs):
        """
        input: probs (bsize, seqlen)
               pred (bsize, seqlen)
        output: confidences (bsize) np.array
        Warning: returns negative perplexity!!!
        """
        confs = []
        for probs_per_ex in probs:
            confs.append(np.log(probs_per_ex).sum()/len(probs_per_ex))
        return np.array(confs)

    def _confidence_ll(self, pred, probs, debug=None):
        """
        input: probs (bsize, seqlen, vocab_size)
        output: confidences (bsize) np.array
        """
        confs = []
        for probs_per_ex in probs:
            #print(probs_per_ex)
            confs.append(np.log(probs_per_ex).sum())
        #print(np.exp(confs))
        return np.array(confs)
    
    def _confidence_beam(self, outputs, output_seqs, probs, threshold=1):
        """
        input: probs (bsize, seqlen, vocab_size)
        output: confidences (bsize) np.array
        """
        confs = []
        model = SimCSE("princeton-nlp/sup-simcse-roberta-base")
        for output_beams, output_seq_beams, prob_beams in zip(outputs, output_seqs, probs):
            #assert len(output_seq_beams) >= 2
            #probs_per_beam = np.array([np.prod(pb[np.arange(len(ob)), ob]) for ob, pb in zip(output_beams, prob_beams)])
            probs_per_beam = np.array([torch.prod(pb.float()) for pb in prob_beams])
            #print(probs_per_beam)
            #import pdb; pdb.set_trace()
            top_beam = output_seq_beams[0]
            sims = np.array([model.similarity(top_beam, other_beam) for other_beam in output_seq_beams[1:]])
            keep_beams = np.hstack([[True], sims > threshold])
            keep_probs = probs_per_beam[keep_beams]
            confs.append(np.log(np.sum(keep_probs)))
        return np.array(confs)
    
    def _confidence_beam_ss(self, scores, output_seqs, threshold=1):
        confs = []
        model = SimCSE("princeton-nlp/sup-simcse-roberta-base")
        for probs_per_beam, output_seq_beams in zip(scores, output_seqs):
            top_beam = output_seq_beams[0]
            sims = np.array([model.similarity(top_beam, other_beam) for other_beam in output_seq_beams[1:]])
            keep_beams = np.hstack([[True], sims > threshold])
            keep_probs = probs_per_beam[keep_beams]
            confs.append(np.exp(np.sum(keep_probs.cpu().numpy())))
        return np.array(confs)

class ConfidenceCalibrator:
    def __init__(self, method, **args):
        self.method = method
        self.c1 = 1 # Scaling of the curve
        self.c2 = -3 # Choose the average LL
        if method == "average_confidence":
            if len(args) != 0:
                self._precompute_sigmoid(**args)
            self.aggregate_confidence = self._aggregate_ac
        elif method == "median_confidence":
            if len(args) != 0:
                self._precompute_sigmoid(**args)
            self.aggregate_confidence = self._aggregate_mc
        elif method == "atc":
            self._precompute_atc(**args)
            self.aggregate_confidence = self._aggregate_atc
        elif method == "curve_fitting":
            self.precompute_fn = self._precompute_cf
            self.aggregate_confidence = self._aggregate_cf
        else:
            raise ValueError

    def __call__(self, confidences):
        #if self.aggregate_confidence is not None:
        #    assert self.fit_params is not None
        return self.aggregate_confidence(confidences)

    def sigmoid(self, x, c1, c2):
        # For negative values
        y = 1 / (1 + np.exp(-c1 * (-c2 + x)))
        return (y)

    def _aggregate_ac(self, confidences):
        rescale = lambda x: self.sigmoid(x, self.c1, self.c2)
        scores = list(map(rescale, confidences))
        return np.mean(scores)
    
    def _aggregate_mc(self, confidences):
        rescale = lambda x: self.sigmoid(x, self.c1, self.c2)
        scores = list(map(rescale, confidences))
        return np.median(scores)
    
    def _aggregate_atc(self, confidences):
        gt_threshold = [x >= self.threshold for x in confidences]
        return sum(gt_threshold)/len(gt_threshold)

    def _aggregate_cf(self, confidences):
        raise NotImplementedError

    def _precompute_sigmoid(self, confidences, corrects):
        # Don't optimize c1 for now
        corrects = [int(x) for x in corrects]
        init = [1, np.median(confidences)]
        popt, pcov = curve_fit(self.sigmoid, confidences, corrects, init, method='dogbox')
        plt.figure()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        plt.tight_layout()
        plt.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        x = np.linspace(-6.0, 0.0, 100)
        y = self.sigmoid(x, self.c1, self.c2)
        ax2.hist(confidences, bins=30, alpha=0.5)
        ax1.scatter(confidences, corrects, alpha=0.5, color='cyan', label='data')
        ax1.plot(x, y, label='sigmoid fit')
        ax1.set_ylabel("Acc")
        ax2.set_ylabel("Freq")
        plt.legend()
        plt.savefig("plots/sigmoid_fit.pdf")
        self.c1, self.c2 = popt

    def _precompute_atc(self, confidences, corrects):
        correct_p = sum(corrects)/len(corrects)
        confidences = sorted(confidences, reverse=True)
        self.threshold = confidences[int(len(confidences) * correct_p)]
    
    def _precompute_cf(self, confidences):
        raise NotImplementedError
