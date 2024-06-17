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


with open("name.txt", "r") as rf:
    pass


def add_detail(filename):
    data = None
    with open(filename, "r") as f:
        data = json.load(f)

    use_data = None
    # with open("ML100k/" + filename, "r") as f:
    #     use_data = json.load(f)

    for i, v in enumerate(data[:]):
        # data[i]["preference"] = use_data[i]["preference"]
        rec_set = set()
        for j in v["result"]:
            rec_set.add(j)

        front_set = set()
        for j in v["front"]:
            front_set.add(j)

        while len(rec_set) < 100:
            index = random.randint(1, 1682)
            if items[index][0] not in front_set:
                rec_set.add(items[index][0])

        rec_list = list(rec_set)
        np.random.shuffle(rec_list)  # 乱序

        data[i]["recommendations"] = rec_list

    with open(filename, "w") as f:
        json.dump(data, f)


add_detail("train.json")
add_detail("test.json")
add_detail("val.json")
