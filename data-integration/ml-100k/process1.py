import pandas as pd
import random
import numpy as np
import json
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
        # print(item)

# with open("name.txt", "w") as wf:
#     for i, v in items.items():
#         wf.write(v[0] + "\n")

# 加载data
header = ["userID", "itemID", "rating", "timestamp"]
data = pd.read_csv("u.data", sep="\t", names=header, engine="python")
# print(data)


class Dataset(object):
    def __init__(self, dataset, u2i=True):
        self.dataset = dataset
        self.use_u2i = u2i
        self.n_users = self.dataset["userID"].nunique()
        self.n_items = self.dataset["itemID"].nunique()
        print("user number:", self.n_users)
        print("item number:", self.n_items)

    def process_data(
        self,
        order=True,
        leave_n=1,
        keep_n=5,
        max_history_length=10,
        premise_threshold=10,
    ):
        # filter ratings by threshold
        self.proc_dataset = self.dataset.copy()
        if order:
            self.proc_dataset = self.proc_dataset.sort_values(
                by=["timestamp", "userID", "itemID"]
            ).reset_index(drop=True)

        self.generate_data_train(history_length=max_history_length, target_length=5)
        # print(self.train_set)

    def generate_data_train(self, history_length, target_length, step=3):
        train_set = []
        test_set = []
        validation_set = []

        processed_data = self.proc_dataset.copy()
        # 数量 29241
        rrr = []
        for uid, group in processed_data.groupby("userID"):  # group by uid
            user_list = []

            ind = len(group.index) - history_length - target_length
            rrr.append(len(group.index))
            while ind >= 0:
                if ind + 30 <= len(group.index):
                    user_list.append(list(group.index[ind : ind + 30]))
                else:
                    if len(group.index) < 30:
                        user_list.append(
                            list(group.index[ind:]) + list(group.index[:ind])
                        )
                    else:
                        user_list.append(
                            list(group.index[ind:])
                            + list(group.index[: 30 - (len(group.index) - ind)])
                        )
                ind -= step
            if len(user_list) > 3:
                # 划分train、test、val
                start = """"""
                prompt = start
                result = []
                front = []
                for index, i in enumerate(user_list[-1]):
                    if index > 9:
                        result.append(items[group.loc[i, "itemID"]][0])
                    else:
                        front.append(items[group.loc[i, "itemID"]][0])
                        v1 = items[group.loc[i, "itemID"]][0]
                        v2 = group.loc[i, "rating"]
                        prompt += f"({v1}, {v2} star); "
                prompt = prompt[:-2] + "."
                validation_set.append(
                    {"history": prompt, "result": result, "front": front}
                )

                prompt = start
                result = []
                front = []
                for index, i in enumerate(user_list[-2]):
                    if index > 9:
                        result.append(items[group.loc[i, "itemID"]][0])
                    else:
                        front.append(items[group.loc[i, "itemID"]][0])
                        v1 = items[group.loc[i, "itemID"]][0]
                        v2 = group.loc[i, "rating"]
                        prompt += f"({v1}, {v2} star); "
                prompt = prompt[:-2] + "."
                test_set.append({"history": prompt, "result": result, "front": front})

                for freq in user_list[:-2]:
                    prompt = start
                    result = []
                    front = []
                    for index, i in enumerate(freq):
                        if index > 9:
                            result.append(items[group.loc[i, "itemID"]][0])
                        else:
                            front.append(items[group.loc[i, "itemID"]][0])
                            v1 = items[group.loc[i, "itemID"]][0]
                            v2 = group.loc[i, "rating"]
                            prompt += f"({v1}, {v2} star); "
                    prompt = prompt[:-2] + "."
                    train_set.append(
                        {"history": prompt, "result": result, "front": front}
                    )

        print("min(rrr)", min(rrr))
        print("max(rrr)", max(rrr))

        print("train len:", len(train_set))
        print("test len:", len(test_set))
        print("val len:", len(validation_set))

        self.validation_set = validation_set
        self.test_set = test_set
        self.train_set = train_set
        with open("train.json", "w") as f:
            json.dump(self.train_set, f)
        with open("test.json", "w") as f:
            json.dump(self.test_set, f)
        with open("val.json", "w") as f:
            json.dump(self.validation_set, f)


ml_data = Dataset(data)
ml_data.process_data()
