import pickle
import os
import random
#import wandb
import uuid
import sys
from tabulate import tabulate
from tqdm import tqdm
import click
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import binned_statistic
from utils import load_all_crossfit
from caching import TaskList, InferenceCache
from config import ontology
import xgboost
from scipy.stats import randint, uniform


class xgb:
    def __init__(self, num_dim:int =20):
        self.vector_dim  = num_dim
        self.model = xgboost.XGBRegressor(objective="reg:squarederror")
        """
        self.parameters = {
                "learning_rate": [ 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
                "lambda": [0, 0.05, 0.1, 0.2, 0.5, 1, 1.2, 1.5]}
        self.cv = GridSearchCV(estimator=self.model, param_grid=self.parameters, cv=5)
        """
        self.param_dist = {
                "learning_rate": uniform(0.01, 0.5),
                "max_depth": randint(3,10),
                "n_estimators": randint(100, 1000),
                'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
                'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
                "gamma": uniform(0,1),
                "reg_alpha": uniform(0,1),
                "reg_lambda": uniform(0,1),
                }
        self.random_search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=self.param_dist,
                n_iter=300,
                cv=5,
                n_jobs=-1,
                verbose=0)

    
    def __call__(self, trainloader, testloader):
        train_X, train_y = np.empty([len(trainloader), self.vector_dim], dtype=object), np.empty(
            [len(trainloader), 1], dtype=object
        )
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            for input_, target_ in zip(inputs, targets):
                train_X[i] = np.array(input_)
                train_y[i] = np.array(target_)
        
        self.random_search.fit(train_X, train_y)
        self.best_params = self.random_search.best_params_
        print(self.best_params)

        self.model = xgboost.XGBRegressor(objective="reg:squarederror", **self.best_params).fit(train_X, train_y)

        test_X, test_y = np.empty([len(testloader), self.vector_dim], dtype=object), np.empty([len(testloader), 1], dtype=object)
        for i, data in enumerate(testloader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            test_X[i] = np.array(inputs)
            test_y[i] = np.array(targets)


        output = self.model.predict(test_X)
        error = np.mean([np.abs(output[idx]-test_y[idx][0]) for idx in range(len(output))])

        return error, np.mean(output)



class KNN:
    def __init__(self, num_neighbor, num_dim=20):
        self.weights = "distance"
        self.neighbor = num_neighbor
        self.model = KNeighborsRegressor(
            n_neighbors=self.neighbor, weights=self.weights
        )
        self.vector_dim=num_dim

    def __call__(self, trainloader, testloader):
        train_X, train_y = np.empty([len(trainloader), self.vector_dim], dtype=object), np.empty(
            [len(trainloader), 1], dtype=object
        )
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            for input_, target_ in zip(inputs, targets):
                train_X[i] = np.array(input_)
                train_y[i] = np.array(target_)
        self.model.fit(train_X, train_y)

        test_X, test_y = np.empty([len(testloader), self.vector_dim], dtype=object), np.empty([len(testloader), 1], dtype=object)
        for i, data in enumerate(testloader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            test_X[i] = np.array(inputs)
            test_y[i] = np.array(targets)

        output = self.model.predict(test_X)
        #neighbors = self.model.kneighbors(test_X)[1].squeeze(1)
        error = np.mean([np.abs(output[idx][0]-test_y[idx][0]) for idx in range(len(output))])

        return error, np.mean(output)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=1536, layers=1, sigmoid=True, dropout=0.0):
        super().__init__()
        if layers == 0:
            modules = [torch.nn.Linear(input_dim, 1),
                    torch.nn.Dropout(dropout)]
        else:
            modules = []
            modules.append(torch.nn.Linear(input_dim, hidden_dim))
            modules.append(torch.nn.Dropout(dropout))
            modules.append(torch.nn.ReLU())
            for i in range(layers - 1):
                modules.append(torch.nn.Linear(hidden_dim, hidden_dim))
                modules.append(torch.nn.Dropout(dropout))
                modules.append(torch.nn.ReLU())
            modules.append(torch.nn.Linear(hidden_dim, 1))
            modules.append(torch.nn.Dropout(dropout))
        if sigmoid:
            modules.append(torch.nn.Sigmoid())
        self.layers = torch.nn.Sequential(*modules)
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        # Danger: not in-place for tensors
        self.to(self.device)


    def forward(self, x):
        return self.layers(x)


    def train_model(self, trainloader, validloader, testloader, lr=1e-4, lr_lambda=0.9, num_epochs=10, run_name="", setting="", layer_input=1):
        """
        train_dataset format:
            List[(distribution (float: data_dim), accuracy (float))]
        """
        self.train()
        #TODO: Fix this wandb project and run naming
        # TODO: Log some config info like train datasets
        #run = wandb.init(reinit=True, project="random", job_type="valid_set", group="13B")
        #run.name = run_name + ":" + run.name
        # TODO: What loss function?
        loss_function = torch.nn.MSELoss()
        valid_criterion = torch.nn.L1Loss()
        # TODO: Maybe do weight decay?
        # TODO: train_batch_size
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch:0.65**epoch)
        best_val_loss = float('inf')
        patience = 7

        train_losses = []
        valid_losses = []
        
        for epoch in range(0, num_epochs):
            train_losses = []
            valid_losses = []
            for i, data in enumerate(trainloader, 0):
                inputs, targets = data
                inputs, targets = inputs.float(), targets.float()
                # .to(device) both inputs & targets
                targets = targets.reshape((targets.shape[0], 1))
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_function(outputs, targets)
                #wandb.log({'train_loss':loss}, step=len(trainloader)*epoch+i)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # Validate
            self.eval()
            for i, data in enumerate(validloader, 0):
                inputs, targets = data
                inputs, targets = inputs.float(), targets.float()

                targets = targets.reshape((targets.shape[0], 1))
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self(inputs)
                loss = valid_criterion(outputs, targets)
                valid_losses.append(loss.item())

                #wandb.log({'valid_loss':loss}, step=len(trainloader)*epoch+i)

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_model = self.state_dict()
                patience = 7
            else:
                patience -= 1
                if patience == 0:
                    print(f"{epoch}/{num_epochs} *** training loss: {round(train_loss,4)}\n valid_loss: {round(valid_loss,4)}")
                    break


            self.train()
            scheduler.step()
        #run.finish()
        self.load_state_dict(best_model)
        losses, all_outputs = self.test_model(testloader)
        return losses, all_outputs


    def test_model(self, testloader, debug=False):
        """
        test_dataset format:
            List[(distribution (float: data_dim), accuracy (float))]
        """
        loss_function = torch.nn.L1Loss()
        losses = []
        all_outputs = []

        for i, data in enumerate(testloader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self(inputs).detach().cpu()
            all_outputs.append(outputs.item())
            loss = loss_function(outputs, targets)
            losses.append(loss.item())
        if debug:
            import pdb; pdb.set_trace()
        return losses, all_outputs

def get_simulated_data(num_samples, num_virtual_samples=2000):
    samples, y = [], []
    for i in range(num_samples):
        true_y = np.random.rand()
        virtual_samples = np.random.normal(
            loc=true_y, scale=0.3, size=num_virtual_samples
        )
        samples.append(virtual_samples)
        y.append(true_y)
    return np.array(samples), np.array(y)


@click.command()
@click.option("--return_error/--return_score", default=True)
@click.option("--setting", type=str, default="cv", help="cross validation. Must pass --cv_k and --tasks")
@click.option("--cv_k", type=int, default=5, help="Number of splits for cross-validation.")
@click.option("--tasks", type=str, default="seq2seq", help="Name of set of tasks in the ontology. Must define in config.py!")
@click.option("--do_sigmoid/--no_sigmoid", type=bool, default=True)
@click.option("--num_unlabeled", type=int, default=1000)
@click.option("--data_dim", type=int, default=0)
@click.option("--dropout", type=float, default=0.2)
@click.option("--lr", type=float, default=1e-5)
@click.option("--lr_lambda", type=float, default=0.8)
@click.option("--num_epochs", type=int, default=50)
@click.option("--only_shots", type=int, default=None)
@click.option("--only_size", type=str, default="llama13B")
@click.option("--seed", type=int, default=1, help="random seed")
@click.option("--train_size", type=int, default=30, help="number of inferences/seed, dataset for training data")
@click.option("--llama/--opt", type=bool, default=True, help="Use LLaMA or OPT")
@click.option("--metric", type=click.Choice(["conf", "pca_embed", "conf_bin", "conf_embed"]), help="metric for dataloader")
def main(
    return_error, 
    setting, 
    do_sigmoid, 
    num_unlabeled, 
    data_dim,
    dropout, 
    lr, 
    lr_lambda, 
    num_epochs, 
    only_shots, 
    only_size,
    seed, 
    cv_k, 
    tasks, 
    train_size,
    llama,
    metric,
):
    if setting == "cv":
        assert cv_k is not None and cv_k > 0 and tasks is not None
    elif setting == "cross_cv":
        assert cv_k is not None and test is not None
    else:
        raise ValueError

    if metric == "pca_embed" or metric == "conf_embed":
        data_dim = 50

    # Setup
    os.environ["WANDB_SILENT"] = "true"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    run_id = str(uuid.uuid4())[:6]
    train_hyperparams = {"do_sigmoid": do_sigmoid, "num_unlabeled": num_unlabeled, "data_dim": data_dim, "dropout": dropout, "lr": lr, "run_id": run_id,
            "lr_lambda": lr_lambda, "num_epochs": num_epochs,"return_error": return_error, "metric": metric, "train_size": train_size}
    
    print(f"Starting Cache!")
    tasks = "seq2seq"
    train_size=30
    all_tasks = TaskList.load_cbqa()
    if llama:
        pkl_path = "pkls_llama_cbqa"
        pca_path = "pkls_llama_cbqa_pca"
    else:
        pkl_path = "pkls_opt_cbqa"
        pca_path = "pkls_opt_cbqa_pca"

    all_inferences = InferenceCache.load_from_path(path=pkl_path, pca_path=pca_path, task_list=all_tasks)
    all_inferences = InferenceCache([inf for inf in all_inferences if inf.mean_score > 0])
    if only_shots:
        all_inferences = all_inferences.of_shots(only_shots)
    if only_size:
        all_inferences = all_inferences.of_size(only_size)
    all_tasks = all_inferences.tasks

    
    print("\n\n\n")
    print(f"Loaded {len(all_tasks)} tasks.")
    print(f"Loaded {len(all_inferences)} pickles.")

    df_headers = []
    results = {}
    if setting == "cv":
        task_names = ontology[tasks]
        all_inferences = all_inferences.of_task(task_names)
        # TODO:
        #   find other ways to filter datasets
        assert cv_k <= len(all_tasks), "Can't do more folds than leave-one-out (n-fold)"
        tasks_cv = all_tasks.split(n=cv_k)
        setting = f"cv_{tasks}"
        print(f"Performing {cv_k}-fold cross-validation on {len(all_inferences)} inferences from {len(all_tasks)} tasks.")
        for test_task_list in tqdm(tasks_cv):
            header = ",".join(test_task_list.names)
            df_headers.append(header)
            test_inferences = all_inferences.of_task(test_task_list)
            # fix the test set: only include first 5 seeds for all shots -> 10 inferences in total for each dataset

            train_inferences = all_inferences.exclude_related(test_inferences)
            train_inferences = train_inferences.top_seeds(train_size)
            results[header] = sweep_methods(train_inferences, test_inferences, setting=setting, **train_hyperparams)


    results = pd.DataFrame(results, columns=df_headers).round(1)
    results = results.dropna(axis="columns", how="all")
    results["mean_accs"] = results.mean(axis="columns")
    results["std"] = results.std(axis="columns")
    # TODO: Print this prettier but for now just get first n cols
    #print(tabulate(results.iloc[:, :7], headers="keys"))
    #order = ['avg_acc', 'std_acc', '1nn', '2nn', '3nn', 'xgb', 'mlp2', 'train_mean', 'avg_conf', 'ATC', '4_lld', '8_lld', '16_lld', '32_lld', '64_lld']
    order = ['avg_acc', 'std_acc', 'ATC', '4_lld', '8_lld', '16_lld', '32_lld', '64_lld']
    results = results.loc[order]
    print(results["mean_accs"], results["std"])


def sweep_methods(
        train_inferences,
        test_inferences,
        setting,
        do_sigmoid,
        num_unlabeled,
        data_dim,
        dropout,
        lr,
        lr_lambda,
        num_epochs,
        return_error,
        run_id,
        metric,
        train_size
):
    results = {}
    """
    for nn in range(1, 4):
        results[f"{nn}nn"] = run_meta_model(
            train=train_inferences,
            test=test_inferences,
            method="knn",
            num_neighbor=nn,
            data_dim=data_dim,
            return_error=return_error,
            num_unlabeled=num_unlabeled,
            setting=setting,
            metric=metric,
        )
    if not return_error:
        results["true_acc"] = run_meta_model(
            train=train_inferences,
            test=test_inferences,
            method="true",
            return_error=False,
            setting=setting,
        )
    results["train_mean"] = run_meta_model(
        train=train_inferences,
        test=test_inferences,
        method="average",
        return_error=return_error,
        num_unlabeled=num_unlabeled,
        setting=setting,
    )
    results["avg_conf"] = run_meta_model(
        train=train_inferences,
        test=test_inferences,
        method="avg_conf",
        return_error=return_error,
        num_unlabeled=num_unlabeled,
        setting=setting
    )
    results["xgb"] = run_meta_model(
        train=train_inferences,
        test=test_inferences,
        method="xgboost",
        data_dim = data_dim,
        return_error=return_error,
        num_unlabeled=num_unlabeled,
        metric=metric,
        setting=setting
    )

    for layers in [2]:
        results[f"mlp{layers}"] = run_meta_model(
            train=train_inferences,
            train_size=train_size,
            test=test_inferences,
            method="mlp",
            return_error=return_error,
            num_unlabeled=num_unlabeled,
            data_dim=data_dim,
            setting=setting,
            layers=layers,
            sigmoid=do_sigmoid,
            dropout=dropout,
            lr=lr,
            lr_lambda=lr_lambda,
            num_epochs=num_epochs,
            run_id=run_id,
            metric=metric,
        )

    results["ATC"] = run_meta_model(
        train=train_inferences,
        test=test_inferences,
        method="ATC",
        return_error=return_error,
        num_unlabeled=num_unlabeled,
        setting=setting)
    """
    results["avg_acc"] = run_meta_model(
        train=train_inferences,
        test=test_inferences,
        method="avg_acc",
        return_error=return_error,
        num_unlabeled=num_unlabeled,
        setting=setting)
        
    results["std_acc"] = run_meta_model(
        train=train_inferences,
        test=test_inferences,
        method="std_acc",
        return_error=return_error,
        num_unlabeled=num_unlabeled,
        setting=setting)
        
    for num_labeled in [4, 8, 16, 32, 64]:
        results[f"{num_labeled}_lld"] = run_meta_model(
            train=train_inferences,
            test=test_inferences,
            method="random",
            return_error=return_error,
            setting=setting,
            num_labeled=num_labeled,
        )
    return results

def run_meta_model(
    train,
    test,
    method,
    train_size=30,
    setting="",
    return_error=False,
    num_unlabeled=500,
    data_dim=20,
    num_labeled=None,
    num_neighbor=None,
    layers=None,
    sigmoid=True,
    dropout=0.0,
    lr=1e-4,
    lr_lambda=0.9,
    num_epochs=10,
    run_id="",
    metric="conf",
):
    # Get Train Dataloader
    train_inferences = train.limit(num_unlabeled) 
    if method == "mlp":
        valid_idx = list(random.sample(range(train_size), int(train_size*0.2)))
        valid_inferences = train_inferences.of_seeds(valid_idx)
        train_inferences = train_inferences.of_seeds(list(set(range(train_size))-set(valid_idx)))
        valid_dataloader = valid_inferences.dataloader(dim=data_dim, batch_size=1, metric=metric)
    # Get Test Dataloader
    test_inferences = test.limit(num_unlabeled)
    train_dataloader = train_inferences.dataloader(dim=data_dim, batch_size=1, metric=metric)
    test_dataloader = test_inferences.dataloader(dim=data_dim, batch_size=1, metric=metric)

    if metric == "conf_embed":
        data_dim += 50
    
    if method == "knn":
        model = KNN(num_neighbor, num_dim=data_dim)
        error, output = model(train_dataloader, test_dataloader)
        if return_error:
            res = error
        else:
            res = output

    elif method == "avg_conf":
        if return_error:
            res = np.abs(np.array(test_inferences.mean_confs) - np.array(test_inferences.mean_scores))
        else:
            res = np.ones((len(test_inferences)) * np.mean(test_inferences.mean_confs))

    elif method == "xgboost":
        model=xgb(num_dim=data_dim)
        error, output = model(train_dataloader, test_dataloader)
        if return_error:
            res=error
        else:
            res=output

    elif method == "ATC":
        pred_list = []
        for test_inf in test_inferences:
            sum_pred = 0
            for train_inf in train_inferences:
                threshold = train_inf.confs[int(len(train_inf.confs)*(1-train_inf.mean_score))]
                sum_pred += len([c for c in test_inf.confs if c<threshold]) / len(test_inf.confs)
            pred_list.append(sum_pred/len(train_inferences))
        if return_error:
            res = np.abs(np.array(test_inferences.mean_scores) - np.array(pred_list))
        else:
            res = np.array(pred_list)

    elif method == "mlp":
        assert layers is not None
        model = MLP(input_dim=data_dim, layers=layers, sigmoid=sigmoid, dropout=dropout)
        test_loss, outputs = model.train_model(train_dataloader, valid_dataloader, test_dataloader, run_name=setting+"_"+str(layers)+"_"+run_id, lr=lr, lr_lambda=lr_lambda, num_epochs=num_epochs, setting=setting, layer_input=layers)
        if return_error:
            res = test_loss
        else:
            res = outputs
    elif method == "true":
        assert not return_error, "Cannot return error of true mean"
        res = [np.mean(acc) for acc in test_inferences.accs]
        #res = np.dot(np.ones(len(test_inferences)),np.mean(test_inferences.y))
    elif method == "average":
        if return_error:
            res = np.abs(np.array(test_inferences.mean_scores) - np.mean(train_inferences.mean_scores)) 
        else:
            res = np.ones(len(test_inferences)) * np.mean(train_inferences.mean_scores)

    elif method ==  "avg_acc":
        res = np.mean(test_inferences.mean_scores) * np.ones(len(test_inferences))

    elif method == "std_acc":
        res = np.std(test_inferences.mean_scores) * np.ones(len(test_inferences))
    
    elif method == "random":
        assert num_labeled is not None
        sample_ys = test_inferences.accs
        true_means = np.array(test_inferences.mean_scores)
        if return_error:
            # TODO: Error actually normalizes out (/100*100). 
            # If using a diff number of samples, change...
            all_errors = []
            for i in range(100):
                estimates = np.array([lld_estimate(list(sample_y), num_labeled) for sample_y in sample_ys])
                error = np.abs(true_means - estimates)
                all_errors.append(error)
            res = np.mean(all_errors, axis=0)
        else:
            res = [lld_estimate(sample_y, num_labeled) for sample_y in sample_ys]
    else:
        raise ValueError
    return np.mean(res) * 100

def lld_estimate(sample_y, num_labeled):
    """
    sample_y (# examples) of f1 scores
    num_labeled: int
    """
    assert isinstance(num_labeled, int)
    if num_labeled > len(sample_y):
        return np.NaN
    else:
        return np.mean(random.sample(sample_y, num_labeled))

if __name__ == "__main__":
    main()
