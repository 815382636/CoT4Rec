import json
import random
import numpy as np
import os


def deal_with(name):
    name = name.strip()
    if name.find("(") != -1:
        name = name[: name.find("(")]
    if name.find("[") != -1:
        name = name[: name.find("[")]
    if name.find(" - ") != -1:
        name = name[: name.find(" - ")]
    # Vol
    if name.find(" Vol") != -1:
        name = name[: name.find(" Vol") - 1]
    #  /
    if name.find(" / ") != -1:
        name = name[: name.find(" / ")]
    name = name.strip()
    if name.endswith("DVD"):
        name = name[:-4]
    if name.endswith("'"):
        name = name[:-1]
    name = name.replace(";", " ")
    name = name.replace("\n", " ")
    name = name.replace("&amp", "&")
    name = name.replace("  ", " ")
    name = name.replace('"', " ")
    name = name.strip()
    return name


meta_data = []
with open("name.txt", "r") as rf:
    for i in rf.readlines():
        meta_data.append(i.replace("\n", ""))
# print("len:", len(meta_data))
# print(meta_data)


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
            index = random.randint(0, len(meta_data) - 1)
            if meta_data[index] not in front_set:
                rec_set.add(meta_data[index])

        rec_list = list(rec_set)
        np.random.shuffle(rec_list)  # 乱序

        data[i]["recommendations"] = rec_list

    with open(filename, "w") as f:
        json.dump(data, f)


# add_detail("train.json")
# add_detail("test.json")
# add_detail("val.json")
