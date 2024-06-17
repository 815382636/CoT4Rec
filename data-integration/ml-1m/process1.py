import pandas as pd
import copy
import json
import random
import numpy as np


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


class Dataset(object):
    def __init__(self, dataset, user_dataset, item_dataset, u2i=True):
        self.dataset = dataset
        self.user_dataset = user_dataset
        self.item_dataset = item_dataset
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
        # self.proc_dataset["rating"][self.proc_dataset["rating"] < threshold] = 0
        # self.proc_dataset["rating"][self.proc_dataset["rating"] >= threshold] = 1
        if order:
            self.proc_dataset = self.proc_dataset.sort_values(
                by=["timestamp", "userID", "itemID"]
            ).reset_index(drop=True)
        self.generate_data_train(history_length=max_history_length, target_length=5)

    # step = 3 train_len: 294892 test_len: 5705  val_len: 5705
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
                    m_name = self.item_dataset.loc[
                        self.item_dataset["itemID"] == group.loc[i, "itemID"]
                    ].values[0][1]
                    m_name = deal_with(m_name)
                    if index > 9:
                        result.append(m_name)
                    else:
                        front.append(m_name)
                        v1 = m_name
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
                    m_name = self.item_dataset.loc[
                        self.item_dataset["itemID"] == group.loc[i, "itemID"]
                    ].values[0][1]
                    m_name = deal_with(m_name)
                    if index > 9:
                        result.append(m_name)
                    else:
                        front.append(m_name)
                        v1 = m_name
                        v2 = group.loc[i, "rating"]
                        prompt += f"({v1}, {v2} star); "
                prompt = prompt[:-2] + "."
                test_set.append({"history": prompt, "result": result, "front": front})

                for freq in user_list[:-2]:
                    prompt = start
                    result = []
                    front = []
                    for index, i in enumerate(freq):
                        m_name = self.item_dataset.loc[
                            self.item_dataset["itemID"] == group.loc[i, "itemID"]
                        ].values[0][1]
                        m_name = deal_with(m_name)
                        if index > 9:
                            result.append(m_name)
                        else:
                            front.append(m_name)
                            v1 = m_name
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


if __name__ == "__main__":

    file = "ratings.dat"
    rating_data = get_ratings(file)
    # print(rating_data)

    file = "movies.dat"
    item_data = get_items(file)
    print(item_data)

    file = "users.dat"
    user_data = get_users(file)
    # print(user_data)

    data = Dataset(rating_data, user_data, item_data)
    data.process_data()
