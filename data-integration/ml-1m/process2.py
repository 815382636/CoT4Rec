import json
import random
import numpy as np
import os
import pandas as pd
import copy


def deal_with(name):
    name = name.strip()
    if name.find("(") != -1:
        name = name[: name.find("(")]
    name = name.strip()
    if name.endswith("The"):
        name = name[:-5]
    if name.endswith("The Movie"):
        name = name[:-11]
    name = name.strip()
    return name


def get_ratings(file):
    header = ["userID", "itemID", "rating", "timestamp"]
    data = pd.read_csv(file, sep="::", names=header, engine="python")
    return data


def get_items(file):
    data = pd.read_csv(
        file,
        engine="python",
        sep="::",
        encoding="ISO-8859-1",
        names=["itemID", "title", "genres"],
    )
    return data


def get_users(file):
    data = pd.read_csv(
        file,
        engine="python",
        sep="::",
        encoding="ISO-8859-1",
        names=["userID", "gender", "age", "occupation", "zip_code"],
    )

    def occupations_map(occupation):
        occupations_dict = {
            1: "technician",
            0: "other",
            2: "writer",
            3: "executive",
            4: "administrator",
            5: "student",
            6: "lawyer",
            7: "educator",
            8: "scientist",
            9: "entertainment",
            10: "programmer",
            11: "librarian",
            12: "homemaker",
            13: "artist",
            14: "engineer",
            15: "marketing",
            16: "none",
            17: "healthcare",
            18: "retired",
            19: "salesman",
            20: "doctor",
        }
        return occupations_dict[occupation]

    data["occupation"] = data["occupation"].apply(
        lambda occupation: occupations_map(occupation)
    )
    data["gender"] = data["gender"].map({"M": "male", "F": "female"})

    # nones = data[data["occupation"] == "none"]
    # data = data.drop(nones.index)

    return data


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
            index = random.randint(0, 3882)
            m_name = deal_with(item_data.loc[index]["title"])
            if m_name not in front_set:
                rec_set.add(m_name)

        rec_list = list(rec_set)
        np.random.shuffle(rec_list)  # 乱序

        data[i]["recommendations"] = rec_list

    print(filename, "len", len(data))

    with open(filename, "w") as f:
        json.dump(data, f)


file = "movies.dat"
item_data = get_items(file)

add_detail("train.json")
add_detail("test.json")
add_detail("val.json")
