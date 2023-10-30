import pickle
import random
import numpy as np
import torch
import utils
import os
from config import ontology
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from caching import TaskList, InferenceCache
from tqdm import tqdm

def retrieve_pca(input_inferences):
    """
    return the pca object fitted to the input_inference embeddings
            the scaler object fitted to the average embeddings across all inferences
    """
    input_embeds = input_inferences.embeds
    avg_embeds = [np.mean(embed, axis=0) for embed in input_embeds]
    embed_arr = np.stack(avg_embeds, axis=0)
    scaler = StandardScaler()
    meta_standard = scaler.fit_transform(embed_arr)
    pca = PCA(n_components=50)
    pca.fit(meta_standard)
    return pca, scaler


def pca_embeddings(model):
    """
    pca transform embeddings (5120,) into (50,) for each inference
    store in pkls_pca/ folder
    """

    all_tasks = TaskList.load_cbqa()
    if model == "llama":
        pkl_path = "pkls_llama_cbqa"
        pca_path = "pkls_llama_cbqa_pca"
    else:
        pkl_path = "pkls_opt_cbqa"
        pca_path = "pkls_opt_cbqa_pca"

    print(f"Starting Cache!")
    all_inferences = InferenceCache.load_from_path(path=pkl_path, task_list=all_tasks)
    print(f"Loaded {len(all_inferences)} pickles.")
    all_inferences = all_inferences.limit(1000)

    if model == "llama":
        for model_size in ["llama7B", "llama13B"]:
            test_inferences = all_inferences.of_size(model_size)
            test_inferences = InferenceCache([inf for inf in test_inferences if inf.mean_score>0])
            test_inferences = InferenceCache([inf for inf in test_inferences if len(np.array(inf.embeds).shape)>1])
            pca_model, scaler = retrieve_pca(test_inferences)

            file_path = pkl_path
            os.makedirs(pca_path, exist_ok=True)
            count = 1

            for fpath in tqdm(os.listdir(file_path)):
                if model_size == "llama7B" and fpath[-6:-4] != "7B":
                    continue
                elif model_size == "llama13B" and fpath[-7:-4] != "13B":
                    continue
                print(f"*** Checking file {fpath} ***")
                full_path = os.path.join(file_path, fpath)
                embed_path = os.path.join(pca_path, fpath)
                if os.path.exists(embed_path):
                    continue
                with open(full_path, "rb") as f:
                    d = pickle.load(f)
                embeds = d["embeds"]
                if not embeds:
                    continue
                
                new_embeds = []
                for embed in embeds:
                    if isinstance(embed, list):
                        new_embeds.append(embed[-1])
                    else:
                        new_embeds.append(embed)
                try:
                    embeds = scaler.transform(np.array(new_embeds))
                except ValueError as e:
                    import pdb; pdb.set_trace()
                new_embeds = pca_model.transform(embeds)
                new_embeds = list(new_embeds)
                with open(embed_path, "wb") as f:
                    pickle.dump(new_embeds, f)
                print(f"*** Complete file {fpath} ***\n")
                count += 1
    else:
        for model_size in ["opt-6.7b", "opt-13b"]:
            test_inferences = all_inferences.of_size(model_size)
            test_inferences = InferenceCache([inf for inf in test_inferences if inf.mean_score>0])
            pca_model, scaler = retrieve_pca(test_inferences)

            file_path = pkl_path
            os.makedirs(pca_path, exist_ok=True)
            count = 1

            for fpath in tqdm(os.listdir(file_path)):
                if model_size == "opt-6.7b" and fpath[-8:-4] != "6.7b":
                    continue
                elif model_size == "opt-13b" and fpath[-7:-4] != "13b":
                    continue
                print(f"*** Checking file {fpath} ***")
                full_path = os.path.join(file_path, fpath)
                embed_path = os.path.join(pca_path, fpath)
                if os.path.exists(embed_path):
                    continue
                with open(full_path, "rb") as f:
                    d = pickle.load(f)
                embeds = d["embeds"]
                embeds = scaler.transform(np.array(embeds))
                new_embeds = pca_model.transform(embeds)
                new_embeds = list(new_embeds)
                with open(embed_path, "wb") as f:
                    pickle.dump(new_embeds, f)
                print(f"*** Complete file {fpath} ***\n")
                count += 1


if __name__ == "__main__":
    for model in ["llama", "opt"]:
        pca_embeddings(model)
        print(f"{model} finished")
