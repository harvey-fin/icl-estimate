import numpy as np
import torch
from torch.nn import functional as F
from torch import nn, optim
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, OPTForCausalLM
from transformers import logging
from simcse import SimCSE
from typing import *
logging.set_verbosity_error()


class OPTForMCQA:
    """
    OPT model for multiple-choice QA tasks
    load from the huggingface opt models https://huggingface.co/facebook/opt-6.7b
    """
    def __init__(self, model_name, max_length=2048, device="cuda", max_new_tokens=None, *args, **kwargs):
        self.max_length = max_length
        self.device = device
        self.max_new_tokens = max_new_tokens
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
            

    def __call__(self, examples, confidence_metric: str = None, label_space:List[str]=["(A)", "(B)", "(C)", "(D)"]):
        self.label_tokens = [self.tokenizer.encode(l)[2] for l in label_space]
        try:
            outputs, probs, embeddings = self.forward(examples)
        except RuntimeError as error:
            raise RuntimeError("Error occurred running the dataset, skipping!")
        
        return outputs, probs, embeddings


    def forward(self, prompts: List[str]) -> Union[List[int], List[np.ndarray]]:
        """
        input: Batch of examples, formatted with prompt included.
        output: List[pred], List[probs]
            outputs: list of output idx
            probs: list of prob float 
        """
        probs = []
        outputs = []
        embeds = []
        for i, prompt in enumerate(prompts):
            try:
                output, prob, embed = self.forward_single([prompt])
            except AssertionError as error:
                raise RuntimeError("Token Length Error occurred")
            if output == -1:
                continue
            outputs.append(output)
            probs.append(prob)
            embeds.append(embed)
            if i%5 == 0:
                print("=", end="")
        print("> completed!")
        return outputs, probs, embeds

    
    def forward_single(self, example: str) ->  Union[int, np.ndarray]:
        """
        input: one example, formatted with prompt included.
        output: pred, probs
            outputs: output idx
            probs: list of prob float 
        """
        inputs = self.tokenizer(
            example,
            return_tensors="pt"
        ).input_ids.to(self.device)
        assert inputs.shape[1] < self.model.config.max_length, "Input token is longer than max length allowed"
        res = self.model.generate(
            inputs,
            output_scores=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            max_new_tokens=self.max_new_tokens,
        )
        hidden_states = res.hidden_states[0][-1][-1,-1,:].cpu().numpy()
        output_token = res.sequences[0][-len(res.scores):]
        for idx, token in enumerate(output_token):
            if token.item() in self.label_tokens:
                return self.label_tokens.index(token.item()), F.softmax(res.scores[idx], dim=-1)[:, self.label_tokens].cpu().numpy()[0], hidden_states
        
        return -1, [], np.ones(1)
