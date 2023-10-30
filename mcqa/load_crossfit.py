import os
import datasets
import numpy as np
from typing import *
from datasets import DatasetDict

labels = ["(A)", "(B)", "(C)", "(D)"]

def load_data(task_name: str) -> DatasetDict:
    """ 
    Load MCQA datasets from crossfit, return dictionary containing 
        question_string: str
        choices: List[str]
        answer_idx: int
    """
    task_name = task_name[9:]
    ret_dict = {}
    if task_name == "wiqa":
        data = datasets.load_dataset("wiqa")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(" ".join(datapoint["question_para_step"]) + " " + datapoint["question_stem"])
            answer_idx = ord(datapoint["answer_label_as_choice"]) - ord("A")
            answers_list.append(answer_idx)
            choices_list.append(datapoint["choices"]["text"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}
        
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["test"]):
            input_list.append(" ".join(datapoint["question_para_step"]) + " " + datapoint["question_stem"])
            answer_idx = ord(datapoint["answer_label_as_choice"]) - ord("A")
            answers_list.append(answer_idx)
            choices_list.append(datapoint["choices"]["text"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

    elif task_name == "wino_grande":
        data = datasets.load_dataset("winogrande", "winogrande_xl")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["sentence"])
            answers_list.append(int(datapoint["answer"])-1)
            choices_list.append([datapoint["option1"], datapoint["option2"]])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}
        
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["sentence"])
            answers_list.append(int(datapoint["answer"])-1)
            choices_list.append([datapoint["option1"], datapoint["option2"]])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "swag":
        data = datasets.load_dataset("swag", "regular")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["startphrase"])
            answers_list.append(datapoint["label"])
            choices_list.append([datapoint["ending0"], datapoint["ending1"], datapoint["ending2"], datapoint["ending3"]])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["startphrase"])
            answers_list.append(datapoint["label"])
            choices_list.append([datapoint["ending0"], datapoint["ending1"], datapoint["ending2"], datapoint["ending3"]])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "superglue-copa":
        data = datasets.load_dataset("super_glue", "copa")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["premise"])
            answers_list.append(datapoint["label"])
            choices_list.append([datapoint["choice1"], datapoint["choice2"]])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}
    
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["premise"])
            answers_list.append(datapoint["label"])
            choices_list.append([datapoint["choice1"], datapoint["choice2"]])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "definite_pronoun_resolution":
        data = datasets.load_dataset("definite_pronoun_resolution")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint, in enumerate(data["train"]):
            input_list.append(datapoint["sentence"] + datapoint["pronoun"] + "refers to")
            choices_list.append(datapoint["candidates"])
            answers_list.append(datapoint["label"])
            if i==99:
                break
        
        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint, in enumerate(data["test"]):
            input_list.append(datapoint["sentence"] + datapoint["pronoun"] + "refers to")
            choices_list.append(datapoint["candidates"])
            answers_list.append(datapoint["label"])
            if i==999:
                break
        
        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}



    elif task_name == "social_i_qa":
        data = datasets.load_dataset("social_i_qa")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["context"] + " " + datapoint["question"])
            answers_list.append(int(datapoint["label"])-1)
            choices_list.append([datapoint["answerA"], datapoint["answerB"], datapoint["answerC"]])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["context"] + " " + datapoint["question"])
            answers_list.append(int(datapoint["label"])-1)
            choices_list.append([datapoint["answerA"], datapoint["answerB"], datapoint["answerC"]])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "race-middle":
        data = datasets.load_dataset("race", "middle")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["article"].replace("\n", " ").replace("\t", " ").replace("\r", " ") + "\n" + datapoint["question"].replace("\n", " ").replace("\t", " ").replace("\r", " "))
            answers_list.append(ord(datapoint["answer"])-ord("A"))
            choices_list.append(datapoint["options"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["test"]):
            input_list.append(datapoint["article"].replace("\n", " ").replace("\t", " ").replace("\r", " ") + "\n" + datapoint["question"].replace("\n", " ").replace("\t", " ").replace("\r", " "))
            answers_list.append(ord(datapoint["answer"])-ord("A"))
            choices_list.append(datapoint["options"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "race-high":
        data = datasets.load_dataset("race", "high")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["article"].replace("\n", " ").replace("\t", " ").replace("\r", " ") + "\n" + datapoint["question"].replace("\n", " ").replace("\t", " ").replace("\r", " "))
            answers_list.append(ord(datapoint["answer"])-ord("A"))
            choices_list.append(datapoint["options"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["test"]):
            input_list.append(datapoint["article"].replace("\n", " ").replace("\t", " ").replace("\r", " ") + "\n" + datapoint["question"].replace("\n", " ").replace("\t", " ").replace("\r", " "))
            answers_list.append(ord(datapoint["answer"])-ord("A"))
            choices_list.append(datapoint["options"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "quartz-with_knowledge":
        data = datasets.load_dataset("quartz", "with_knowledge")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["para"] + " " + datapoint["question"])
            answers_list.append(datapoint["choices"]["label"].index(datapoint["answerKey"]))
            choices_list.append(datapoint["choices"]["text"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["test"]):
            input_list.append(datapoint["para"] + " " + datapoint["question"])
            answers_list.append(datapoint["choices"]["label"].index(datapoint["answerKey"]))
            choices_list.append(datapoint["choices"]["text"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "quartz-no_knowledge":
        data = datasets.load_dataset("quartz", "no_knowledge")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["question"])
            answers_list.append(datapoint["choices"]["label"].index(datapoint["answerKey"]))
            choices_list.append(datapoint["choices"]["text"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["test"]):
            input_list.append(datapoint["question"])
            answers_list.append(datapoint["choices"]["label"].index(datapoint["answerKey"]))
            choices_list.append(datapoint["choices"]["text"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "quarel":
        data = datasets.load_dataset("quarel")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            st1 = datapoint["question"].find("(A)")
            st2 = datapoint["question"].find("(B)")
            input_list.append(datapoint["question"][:st1])
            answers_list.append(datapoint["answer_index"])
            choices_list.append([datapoint["question"][st1+4:st2], datapoint["question"][st2+4:]])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["test"]):
            st1 = datapoint["question"].find("(A)")
            st2 = datapoint["question"].find("(B)")
            input_list.append(datapoint["question"][:st1])
            answers_list.append(datapoint["answer_index"])
            choices_list.append([datapoint["question"][st1+4:st2], datapoint["question"][st2+4:]])
            if i==99:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "qasc":
        data = datasets.load_dataset("qasc")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["combinedfact"] + " " + datapoint["question"])
            answers_list.append(datapoint["choices"]["label"].index(datapoint["answerKey"]))
            choices_list.append(datapoint["choices"]["text"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["combinedfact"] + " " + datapoint["question"])
            answers_list.append(datapoint["choices"]["label"].index(datapoint["answerKey"]))
            choices_list.append(datapoint["choices"]["text"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "openbookqa":
        data = datasets.load_dataset("openbookqa", "main")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["question_stem"])
            answers_list.append(datapoint["choices"]["label"].index(datapoint["answerKey"]))
            choices_list.append(datapoint["choices"]["text"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        data = datasets.load_dataset("openbookqa", "main")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["test"]):
            input_list.append(datapoint["question_stem"])
            answers_list.append(datapoint["choices"]["label"].index(datapoint["answerKey"]))
            choices_list.append(datapoint["choices"]["text"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "hellaswag":
        data = datasets.load_dataset("hellaswag")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["ctx"])
            answers_list.append(int(datapoint["label"]))
            choices_list.append(datapoint["endings"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["ctx"])
            answers_list.append(int(datapoint["label"]))
            choices_list.append(datapoint["endings"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "dream":
        data = datasets.load_dataset("dream")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(" ".join(datapoint["dialogue"]) + " " + datapoint["question"])
            answers_list.append(datapoint["choice"].index(datapoint["answer"]))
            choices_list.append(datapoint["choice"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["test"]):
            input_list.append(" ".join(datapoint["dialogue"]) + " " + datapoint["question"])
            answers_list.append(datapoint["choice"].index(datapoint["answer"]))
            choices_list.append(datapoint["choice"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "cosmos_qa":
        data = datasets.load_dataset("cosmos_qa")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["context"] + " " + datapoint["question"])
            choices_list.append([datapoint["answer0"], datapoint["answer1"], datapoint["answer2"], datapoint["answer3"]])
            answers_list.append(datapoint["label"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["context"] + " " + datapoint["question"])
            choices_list.append([datapoint["answer0"], datapoint["answer1"], datapoint["answer2"], datapoint["answer3"]])
            answers_list.append(datapoint["label"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "commonsense_qa":
        data = datasets.load_dataset("commonsense_qa")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["question"])
            answers_list.append(datapoint["choices"]["label"].index(datapoint["answerKey"]))
            choices_list.append(datapoint["choices"]["text"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["question"])
            answers_list.append(datapoint["choices"]["label"].index(datapoint["answerKey"]))
            choices_list.append(datapoint["choices"]["text"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "ai2_arc":
        data = datasets.load_dataset("ai2_arc", "ARC-Challenge")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["question"])
            answers_list.append(datapoint["choices"]["label"].index(datapoint["answerKey"]))
            choices_list.append(datapoint["choices"]["text"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["test"]):
            input_list.append(datapoint["question"])
            answers_list.append(datapoint["choices"]["label"].index(datapoint["answerKey"]))
            choices_list.append(datapoint["choices"]["text"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "codah":
        data = datasets.load_dataset("codah", "fold_0")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["question_propmt"])
            answers_list.append(datapoint["correct_answer_idx"])
            choices_list.append(datapoint["candidate_answers"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["test"]):
            input_list.append(datapoint["question_propmt"])
            answers_list.append(datapoint["correct_answer_idx"])
            choices_list.append(datapoint["candidate_answers"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "aqua_rat":
        data = datasets.load_dataset("aqua_rat", "raw")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["question"])
            for i, c in enumerate(["A", "B", "C", "D", "E"]):
                if c==datapoint["correct"]:
                    answers_list.append(i)
            choices_list.append([ans[2:] for ans in datapoint["options"]])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["test"]):
            input_list.append(datapoint["question"])
            for i, c in enumerate(["A", "B", "C", "D", "E"]):
                if c==datapoint["correct"]:
                    answers_list.append(i)
            choices_list.append([ans[2:] for ans in datapoint["options"]])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "quail":
        data = datasets.load_dataset("quail")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["context"] + " " + datapoint["question"])
            answers_list.append(datapoint["correct_answer_id"])
            choices_list.append(datapoint["answers"])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["validation"]):
            input_list.append(datapoint["context"] + " " + datapoint["question"])
            answers_list.append(datapoint["correct_answer_id"])
            choices_list.append(datapoint["answers"])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    elif task_name == "math_qa":
        data = datasets.load_dataset("math_qa")
        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["train"]):
            input_list.append(datapoint["Problem"])
            for i, c in enumerate(["a", "b", "c", "d", "e"]):
                if c==datapoint["correct"]:
                    answers_list.append(i)
            choices_list.append([ans[4:] for ans in datapoint["options"]])
            if i==99:
                break

        ret_dict["train"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}

        input_list = []
        choices_list = []
        answers_list = []
        for i, datapoint in enumerate(data["test"]):
            input_list.append(datapoint["Problem"])
            for i, c in enumerate(["a", "b", "c", "d", "e"]):
                if c==datapoint["correct"]:
                    answers_list.append(i)
            choices_list.append([ans[4:] for ans in datapoint["options"].split(" , ")])
            if i==999:
                break

        ret_dict["test"] = {"questions": input_list, "choices": choices_list, "answers": answers_list}


    return DatasetDict(ret_dict)

if __name__ == "__main__":
    data = load_data("crossfit:math_qa")["test"]
