import json
import random
import numpy as np


name_change = dict()
name_list = None
with open("all_name.json", "r") as rf:
    name_list = json.load(rf)
    for i in name_list:
        name_change[i["source"]] = i["target"]


def deal_with(name):
    new_name = name_change.get(name) if name_change.get(name) else name
    if "." in new_name:
        new_name = new_name[: new_name.find(".")]
    if "(" in new_name:
        new_name = new_name[: new_name.find("(")]
    if "," in new_name:
        new_name = new_name[: new_name.find(",")]
    return new_name


def add_detail(filename):
    data = None
    with open(filename, "r") as f:
        data = json.load(f)

    for i, v in enumerate(data[:]):
        rec_set = set()
        for j in v["result"]:
            rec_set.add(j)

        front_set = set()
        for j in v["front"]:
            front_set.add(j)

        while len(rec_set) < 100:
            index = random.randint(0, len(name_list) - 1)
            if name_list[index]["target"] not in front_set:
                rec_set.add(name_list[index]["target"])

        rec_list = list(rec_set)
        np.random.shuffle(rec_list)  # 乱序

        data[i]["recommendations"] = rec_list

    with open(filename, "w") as f:
        json.dump(data, f)


add_detail("train.json")
add_detail("test.json")
add_detail("val.json")
