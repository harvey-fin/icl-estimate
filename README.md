# Estimating Large Language Model Capabilities without Labeled Test Data

This paper is the code implementation of "[Estimating Large Language Model Capabilities without Labeled Test Data](https://arxiv.org/abs/2305.14802)" by [Harvey Yiyun Fu](https://harvey-fin.github.io/), [Qinyuan Ye](http://yeqy.xyz/), [Albert Xu](https://scholar.google.com/citations?user=iFeGf_EAAAAJ&hl=en), [Xiang Ren](https://shanzhenren.github.io/), and [Robin Jia](https://robinjia.github.io/).

This repo contains code for both the In-context Learning LLM inferece to generate meta-training data and the meta-model training. 

## Installation
-----

```
pip install torch==1.13.1
pip install transformers==4.22.1
```

## Quick Start
-----

To run experiments with MMLU and MCQA datasets:
```
cd mcqa
```

To run experiments with CBQA datasets:
```
cd cbqa
```
Please see `mcqa/config.py` and `cbqa/config.py` for the full ontology of datasets in each collection. 


### Meta-training data collection
-----

While under `mcqa/`, run the following command to do inference using the OPT model on the `MMLU/MCQA` dataset:
```
python opt_mmlu_worker.py\
    --model_size opt-6.7b --num_shots 5 --temperature 0 --template mmlu --seed 1
```
- `--model_size`: size of the OPT model, such as `opt-6.7b` or `opt-13b`
- `--num_shots`: number of few-shot examples in the prompt
- `--template`: the prompt template to demonstrate the few-shot examples. Choose from `mmlu`, `subject`, `gopher`, `gpt`, and `user`.
- `--temperature`: hyperparameter to control the randomness
- `--seed`: random seed

Similarly, while under `cbqa/`, run the following command to do inference using the OPT model on the `CBQA` dataset:
```
python opt_worker.py\
    --model_size opt-6.7b --num_shots 5 --seed 1
```
- `--model_size`: size of the OPT model, such as `opt-6.7b` or `opt-13b`
- `--num_shots`: number of few-shot examples in the prompt
- `--seed`: random seed

Then under either directory, run
```
python transform_embed.py
```
to retrieve and store the PCA transformed embeddings


### Meta-model training
-----
Under `mcqa/` , run
```
python train_classifier.py\
    --setting cv --cv_k 5 --tasks mmlu --num_unlabeled 1000 --data_dim 100 --only_size 13B \
    --seed 1 --llama --mmlu --metric conf
```

- `--setting`: general setting for train/test split
- `--cv_k`: number of splits for cross validation
- `--tasks`: task defined in `config.py` as the meta data, choose from `mmlu` and `mcqa`
- `--num_unlabeled`: how much data to include in a single confidence profile
- `--data_dim`: dimension of confidence profile
- `--only_size`: inference of the specified size of LLM
- `--only_shots`: inference of the specified k-shot results
- `--llama/--opt`: use llama model or opt model
- `--mmlu/--mcqa`: do inference on `mmlu` or `mcqa`
- `--metric`: metric for processing confidence profile, choose from `conf`, `pca_embed`, `conf_embed`
- `--train_size`: number of seed to include in the training/test data
- `--seed`: random seed
- `--do_sigmoid`, `--dropout`, `--lr`, `--lr_lambda`, `--num_epochs`: MLP hyperparameters

Under `cbqa/`, run 
```
python train_classifier.py\
    --setting cv --cv_k 5 --tasks cbqa --num_unlabeled 1000 --data_dim 100 --only_size llama13B \
    --seed 1 --llama --metric conf
```

- `--setting`: general setting for train/test split
- `--cv_k`: number of splits for cross validation
- `--tasks`: task defined in `config.py` as the meta data, choose from `cbqa` and `seq2seq`
- `--num_unlabeled`: how much data to include in a single confidence profile
- `--data_dim`: dimension of confidence profile 
- `--only_size`: inference of the specified size of LLM
- `--only_shots`: inference of the specified k-shot results
- `--llama/--opt`: use llama model or opt model
- `--mmlu/--mcqa`: do inference on `mmlu` or `mcqa`
- `--metric`: metric for processing confidence profile, choose from `conf`, `pca_embed`, `conf_embed`
- `--seed`: random seed 
- `--do_sigmoid`, `--dropout`, `--lr`, `--lr_lambda`, `--num_epochs`: MLP hyperparameters

## Acknowledgement
-----
We did not include meta-training data in this repo due to its large magnitude. We did not include the inference code for LLaMA models to avoid certain copyright issues. We thank :hugs: [huggingface datasets](https://github.com/huggingface/datasets) for making the datasets and LLMs easily accessible.





