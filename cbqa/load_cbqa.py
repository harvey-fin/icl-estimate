import os
import datasets
import numpy as np
import random
from datasets import DatasetDict
from config import ontology
from typing import *

random.seed(42)

def load_data(task_name: str) -> DatasetDict:
    """
    Load CBQA datasets from crossfit, return dictionary containing
        question_string: str
        answer: str
    """

    task_name = task_name[9:]
    ret_dict = {}
    
    if task_name == "squad-no_context":
        data = datasets.load_dataset("squad")
        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["question"])
            answers_list.append(datapoint["answers"]["text"][0])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["question"])
            answers_list.append(datapoint["answers"]["text"][0])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}


    if task_name == "squad-with_context":
        data = datasets.load_dataset("squad")
        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["context"] + " " + datapoint["question"])
            answers_list.append(datapoint["answers"]["text"][0])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["context"] + " " + datapoint["question"])
            answers_list.append(datapoint["answers"]["text"][0])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}


    if task_name == "numer_sense":
        data = datasets.load_dataset("numer_sense")
        input_list = []
        answers_list = []
        train_idx = random.sample(range(4000), 99)
        for idx in train_idx:
            input_list.append(data["train"][idx]["sentence"].replace("<mask>", "[MASK]"))
            answers_list.append(data["train"][idx]["target"])

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        test_idx = random.sample(range(4000, 10000), 999)
        for idx in test_idx:
            input_list.append(data["train"][idx]["sentence"].replace("<mask>", "[MASK]"))
            answers_list.append(data["train"][idx]["target"])

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}


    if task_name == "kilt_trex":
        data = datasets.load_dataset("kilt_tasks", "trex")
        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["input"])
            answers_list.append(datapoint["output"][0]["answer"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["input"])
            answers_list.append(datapoint["output"][0]["answer"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}


    if task_name == "kilt_hotpotqa":
        data = datasets.load_dataset("kilt_tasks", "hotpotqa")
        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["input"])
            answers_list.append(datapoint["output"][0]["answer"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["input"])
            answers_list.append(datapoint["output"][0]["answer"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}


    if task_name == "kilt_nq":
        data = datasets.load_dataset("kilt_tasks", "nq")
        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["input"])
            answers_list.append(datapoint["output"][0]["answer"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["input"])
            answers_list.append(datapoint["output"][0]["answer"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}


    if task_name == "kilt_zsre":
        data = datasets.load_dataset("kilt_tasks", "structured_zeroshot")
        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["input"])
            answers_list.append(datapoint["output"][0]["answer"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["input"])
            answers_list.append(datapoint["output"][0]["answer"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}


    if task_name == "lama-trex":
        data = datasets.load_dataset("lama", "trex")
        input_list = []
        answers_list = []
        train_idx = random.sample(range(800000), 99)
        for idx in train_idx:
            input_list.append(data["train"][idx]["template"].replace("[X]", data["train"][idx]["sub_label"]).replace("[Y]", "[MASK]"))
            answers_list.append(data["train"][idx]["obj_label"])

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        test_idx = random.sample(range(800000, 1000000), 999)
        for idx in test_idx:
            input_list.append(data["train"][idx]["template"].replace("[X]", data["train"][idx]["sub_label"]).replace("[Y]", "[MASK]"))
            answers_list.append(data["train"][idx]["obj_label"])

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}


    if task_name == "lama-squad":
        data = datasets.load_dataset("lama", "squad")
        input_list = []
        answers_list = []
        train_idx = range(32)
        for idx in train_idx:
            input_list.append(data["train"][idx]["masked_sentence"])
            answers_list.append(data["train"][idx]["obj_label"])

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        test_idx = range(32, 300)
        for idx in test_idx:
            input_list.append(data["train"][idx]["masked_sentence"])
            answers_list.append(data["train"][idx]["obj_label"])

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}

    
    if task_name == "lama-google_re":
        data = datasets.load_dataset("lama", "google_re")
        input_list = []
        answers_list = []
        train_idx = random.sample(range(1000), 99)
        for idx in train_idx:
            input_list.append(data["train"][idx]["template"].replace("[X]", data["train"][idx]["sub_label"]).replace("[Y]", "[MASK]"))
            answers_list.append(data["train"][idx]["obj_label"])

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        test_idx = random.sample(range(1000, 6000), 999)
        for idx in test_idx:
            input_list.append(data["train"][idx]["template"].replace("[X]", data["train"][idx]["sub_label"]).replace("[Y]", "[MASK]"))
            answers_list.append(data["train"][idx]["obj_label"])

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}


    if task_name == "lama-conceptnet":
        data = datasets.load_dataset("lama", "conceptnet")
        input_list = []
        answers_list = []
        train_idx = random.sample(range(2000), 99)
        for idx in train_idx:
            input_list.append(data["train"][idx]["masked_sentence"])
            answers_list.append(data["train"][idx]["obj_label"])

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        test_idx = random.sample(range(2000, 10000), 999)
        for idx in test_idx:
            input_list.append(data["train"][idx]["masked_sentence"])
            answers_list.append(data["train"][idx]["obj_label"])

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}


    if task_name == "freebase_qa":
        data = datasets.load_dataset("freebase_qa")
        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["RawQuestion"])
            answers_list.append(datapoint["Parses"]["Answers"][0]["AnswersName"][0][0])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["test"]):
            input_list.append(datapoint["RawQuestion"])
            answers_list.append(datapoint["Parses"]["Answers"][0]["AnswersName"][0][0])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}


    if task_name == "web_questions":
        data = datasets.load_dataset("web_questions")
        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["question"])
            answers_list.append(datapoint["answers"][0])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["test"]):
            input_list.append(datapoint["question"])
            answers_list.append(datapoint["answers"][0])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}


    if task_name == "jeopardy":
        data = datasets.load_dataset("jeopardy")
        input_list = []
        answers_list = []
        train_idx = random.sample(range(100000), 99)
        for idx in train_idx:
            input_list.append("category: " + data["train"][idx]["category"] + " question: " + data["train"][idx]["question"])
            answers_list.append(data["train"][idx]["answer"])

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        test_idx = random.sample(range(100000, 200000), 999)
        for idx in test_idx:
            input_list.append("category: " + data["train"][idx]["category"] + " question: " + data["train"][idx]["question"])
            answers_list.append(data["train"][idx]["answer"])

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}


    if task_name == "spider":
        data = datasets.load_dataset("spider")
        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["question"])
            answers_list.append(datapoint["query"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["question"])
            answers_list.append(datapoint["query"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}


    if task_name == "ade_corpus_v2-dosage":
        data = datasets.load_dataset("ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation")
        input_list = []
        answers_list = []
        train_idx = random.sample(range(1000), 99)
        for idx in train_idx:
            input_list.append(data["train"][idx]["text"] + " the effect of " + data["train"][idx]["drug"] + " is")
            answers_list.append(data["train"][idx]["effect"])

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        test_idx = random.sample(range(1000, 6000), 999)
        for idx in test_idx:
            input_list.append(data["train"][idx]["text"] + " the effect of " + data["train"][idx]["drug"] + " is")
            answers_list.append(data["train"][idx]["effect"])

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}


    if task_name == "wikisql":
        data = datasets.load_dataset("wikisql")
        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["question"])
            answers_list.append(datapoint["sql"]["human_readable"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["test"]):
            input_list.append(datapoint["question"])
            answers_list.append(datapoint["sql"]["human_readable"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}


    if task_name == "gigaword":
        data = datasets.load_dataset("gigaword")
        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["document"])
            answers_list.append(datapoint["summary"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["test"]):
            input_list.append(datapoint["document"])
            answers_list.append(datapoint["summary"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}

    
    if task_name == "adversarialqa":
        data = datasets.load_dataset("adversarial_qa", "adversarialQA")
        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["context"] + " " + datapoint["question"])
            answers_list.append(datapoint["answers"]["text"][0])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["context"] + " " + datapoint["question"])
            answers_list.append(datapoint["answers"]["text"][0])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}

    
    if task_name == "ropes":
        data = datasets.load_dataset("ropes")
        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["background"] + " " + datapoint["situation"] + " " + datapoint["question"])
            answers_list.append(datapoint["answers"]["text"][0])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["background"] + " " + datapoint["situation"] + " " + datapoint["question"])
            answers_list.append(datapoint["answers"]["text"][0])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}


    if task_name == "tweet_qa":
        data = datasets.load_dataset("tweet_qa")
        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["Tweet"] + " " + datapoint["Question"])
            answers_list.append(datapoint["Answer"][0])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "answers": answers_list}

        input_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["Tweet"] + " " + datapoint["Question"])
            answers_list.append(datapoint["Answer"][0])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "answers": answers_list}


    return DatasetDict(ret_dict)



if __name__ == "__main__":
    for task in ontology["seq2seq"]:
        data = load_data(task)["test"]
        print(data["questions"][:2])
        print(data["answers"][:2])
        import pdb; pdb.set_trace()




































