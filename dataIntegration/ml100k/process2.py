import json
import random
import numpy as np
import os


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


# 加载user
users = dict()
with open("u.user", "r") as rf:
    for i in rf.readlines():
        i = i.strip()
        if not i or i == "":
            continue
        load_user = i.split("|")
        for j, v in enumerate(load_user[:]):
            load_user[j] = v.strip()
            if load_user[j] == "M":
                load_user[j] = "male"
            if load_user[j] == "F":
                load_user[j] = "female"
        users[int(load_user[0])] = load_user
        # print(load_user)

# 加载genre
genre = dict()
with open("u.genre", "r") as rf:
    for i in rf.readlines():
        i = i.strip()
        if not i or i == "":
            continue
        load_genre = i.split("|")
        genre[load_genre[1]] = load_genre[0]

# 加载item
items = dict()
with open("u.item", "r") as rf:
    for i in rf.readlines():
        i = i.strip()
        if not i or i == "":
            continue
        load_item = i.split("|")
        # 构建genre
        item = [load_item[1]]
        genre_str = ""
        for j in range(18):
            if load_item[-18 + j] == "1":
                genre_str = genre_str + genre[str(j)] + "|"
        genre_str = genre_str[:-1]
        item.append(genre_str)
        item[0] = deal_with(item[0])
        items[int(load_item[0])] = item
# print(items)


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


# add_detail("train.json")
# add_detail("test.json")
# add_detail("val.json")
