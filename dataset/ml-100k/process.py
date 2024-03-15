import pandas as pd
import random
import numpy as np
import json

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
        items[int(load_item[0])] = item
        # print(item)

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
        max_history_length=5,
        premise_threshold=0,
    ):
        # filter ratings by threshold
        self.proc_dataset = self.dataset.copy()
        if order:
            self.proc_dataset = self.proc_dataset.sort_values(
                by=["timestamp", "userID", "itemID"]
            ).reset_index(drop=True)

        self.leave_one_out_by_time(leave_n, keep_n)
        self.generate_histories(max_hist_length=max_history_length, premise_threshold=0)
        # print(self.train_set)

    def leave_one_out_by_time(self, leave_n=1, keep_n=5):
        train_set = []
        # generate training set by looking for the first keep_n POSITIVE interactions
        processed_data = self.proc_dataset.copy()
        for uid, group in processed_data.groupby("userID"):  # group by uid
            found, found_idx = 0, -1
            for idx in group.index:
                if group.loc[idx, "rating"] > 3:
                    found_idx = idx
                    found += 1
                    if found >= keep_n:
                        break
            if found_idx > 0:
                train_set.append(group.loc[:found_idx])
        train_set = pd.concat(train_set)
        # drop the training data info
        processed_data = processed_data.drop(train_set.index)

        # generate test set by looking for the last leave_n POSITIVE interactions
        test_set = []
        for uid, group in processed_data.groupby("userID"):
            found, found_idx = 0, -1
            for idx in reversed(group.index):
                if group.loc[idx, "rating"] > 3:
                    found_idx = idx
                    found += 1
                    if found >= leave_n:
                        break
            if found_idx > 0:
                test_set.append(group.loc[found_idx:])
        test_set = pd.concat(test_set)
        processed_data = processed_data.drop(test_set.index)

        validation_set = []
        for uid, group in processed_data.groupby("userID"):
            found, found_idx = 0, -1
            for idx in reversed(group.index):
                if group.loc[idx, "rating"] > 3:
                    found_idx = idx
                    found += 1
                    if found >= leave_n:
                        break
            # put all the negative interactions encountered during the search process into validation set
            if found_idx > 0:
                validation_set.append(group.loc[found_idx:])
        validation_set = pd.concat(validation_set)
        processed_data = processed_data.drop(validation_set.index)

        # The remaining data (after removing validation and test) are all in training data
        self.train_set = pd.concat([train_set, processed_data])
        self.validation_set, self.test_set = validation_set.reset_index(
            drop=True
        ), test_set.reset_index(drop=True)

    def generate_histories(self, max_hist_length=5, premise_threshold=0):
        # TODO insert an assert to check the value of the parameter premise_threshold (read doc)
        history_dict = (
            {}
        )  # it contains for each user the list of all the items he has seen
        feedback_dict = (
            {}
        )  # it contains for each user the list of feedbacks he gave to the items he has seen
        for df in [self.train_set, self.validation_set, self.test_set]:
            history = (
                []
            )  # each element of this list is a list containing the history items of a single interaction
            fb = (
                []
            )  # each element of this list is a list containing the feedback for the history items of a
            # single interaction
            hist_len = (
                []
            )  # each element of this list indicates the number of history items of a single interaction
            type = []  # 2表示u2i节点， 1表示i2u
            uids, iids, feedbacks = (
                df["userID"].tolist(),
                df["itemID"].tolist(),
                df["rating"].tolist(),
            )
            for i, uid in enumerate(uids):
                iid, feedback = iids[i], feedbacks[i]

                if uid not in history_dict:
                    history_dict[uid] = []
                    feedback_dict[uid] = []

                # list containing the history for current interaction
                tmp_his = (
                    copy.deepcopy(history_dict[uid])
                    if max_hist_length == 0
                    else history_dict[uid][-max_hist_length:]
                )
                # list containing the feedbacks for the history of current interaction
                fb_his = (
                    copy.deepcopy(feedback_dict[uid])
                    if max_hist_length == 0
                    else feedback_dict[uid][-max_hist_length:]
                )

                history.append(tmp_his)
                fb.append(fb_his)
                hist_len.append(len(tmp_his))
                if tmp_his == []:
                    type.append(2)
                else:
                    type.append(4)
                history_dict[uid].append(iid)
                feedback_dict[uid].append(feedback)

            df["history"] = history
            df["history_feedback"] = fb
            df["history_length"] = hist_len
            df["type"] = type

        if premise_threshold != 0:
            self.train_set = self.train_set[
                self.train_set.history_length > premise_threshold
            ]
            self.validation_set = self.validation_set[
                self.validation_set.history_length > premise_threshold
            ]
            self.test_set = self.test_set[
                self.test_set.history_length > premise_threshold
            ]

        self.clean_data()

    def clean_data(self):
        train_type1 = self.train_set.copy()
        train_type1["type"] = 1
        type1 = train_type1.sample(frac=1.0, replace=False, random_state=2022, axis=0)
        self.train_set = self.train_set[self.train_set["rating"] > 3].reset_index(
            drop=True
        )
        self.train_set = self.train_set[
            self.train_set["history_feedback"].map(len) >= 1
        ].reset_index(drop=True)
        self.validation_set = self.validation_set[
            self.validation_set["rating"] > 3
        ].reset_index(drop=True)
        self.validation_set = self.validation_set[
            self.validation_set["history_feedback"].map(len) >= 1
        ].reset_index(drop=True)
        self.test_set = self.test_set[self.test_set["rating"] > 3].reset_index(
            drop=True
        )
        self.test_set = self.test_set[
            self.test_set["history_feedback"].map(len) >= 1
        ].reset_index(drop=True)
        if self.use_u2i:
            self.train_set = pd.concat(
                [type1, self.train_set], axis=0, ignore_index=True
            )

    def generate_train_prompt(self, select_num=4):

        train_prompts = []
        for index, row in self.train_set.iterrows():
            user = users[row["userID"]]
            insert = {}

            gender = user[2]
            age = user[1]
            occupation = user[3]

            if gender == "female":
                prompt = f"""Given a user who is '{gender}, {age} years old, and occupation is {occupation}', her movie viewing history over time includes: """
            else:
                prompt = f"""Given a user who is '{gender}, {age} years old, and occupation is {occupation}', his movie viewing history over time includes: """

            for i, v in enumerate(row["history"]):
                item = items[v]
                # 存在无打分情况？
                if len(row["history_feedback"]) > i:
                    item_scale = row["history_feedback"][i]
                else:
                    item_scale = 3
                prompt += f"({item[0]}, {item[1]}, {item_scale} star); "
            prompt = prompt[:-2] + "."
            insert["prompt"] = prompt

            value = items[row["itemID"]]

            cot_prompt = f"""Please briefly describe the user's viewing preferences and briefly analyze the reasons why the user decides to watch {value[0]} ({value[1]}) next. (within 200 words)"""

            insert["cot_prompt"] = cot_prompt
            insert["value"] = value[0] + ", " + value[1]
            # insert["value_scale"] = row["rating"]

            # 生成随机选项
            exist_movies = row["history"]
            exist_movies.append(row["itemID"])
            exist_movies = set(exist_movies)
            select = [value[0] + ", " + value[1]]
            while len(select) < select_num:
                num = random.randint(1, self.n_items)
                if num not in exist_movies:
                    item = items[num]
                    if len(item) > 0:
                        select.append(item[0] + ", " + item[1])
                        exist_movies.add(num)
            np.random.shuffle(select)  # 乱序
            insert["select"] = select

            train_prompts.append(insert)
            if index % 1000 == 0:
                print(index)

        print("train len:", len(train_prompts))
        with open("train.json", "w") as f:
            json.dump(train_prompts, f)

    def generate_test_prompt(self, select_num=4):

        train_prompts = []
        for index, row in self.test_set.iterrows():
            user = users[row["userID"]]
            insert = {}

            gender = user[2]
            age = user[1]
            occupation = user[3]

            if gender == "female":
                prompt = f"""Given a user who is '{gender}, {age} years old, and occupation is {occupation}', her movie viewing history over time includes: """
            else:
                prompt = f"""Given a user who is '{gender}, {age} years old, and occupation is {occupation}', his movie viewing history over time includes: """

            for i, v in enumerate(row["history"]):
                item = items[v]
                # 存在无打分情况？
                if len(row["history_feedback"]) > i:
                    item_scale = row["history_feedback"][i]
                else:
                    item_scale = 3
                prompt += f"({item[0]}, {item[1]}, {item_scale} star); "
            prompt = prompt[:-2] + "."
            insert["prompt"] = prompt

            value = items[row["itemID"]]

            cot_prompt = f"""Please briefly describe the user's viewing preferences and briefly analyze the reasons why the user decides to watch {value[0]} ({value[1]}) next. (within 200 words)"""

            insert["cot_prompt"] = cot_prompt
            insert["value"] = value[0] + ", " + value[1]
            # insert["value_scale"] = row["rating"]

            # 生成随机选项
            exist_movies = row["history"]
            exist_movies.append(row["itemID"])
            exist_movies = set(exist_movies)
            select = [value[0] + ", " + value[1]]
            while len(select) < select_num:
                num = random.randint(1, self.n_items)
                if num not in exist_movies:
                    item = items[num]
                    if len(item) > 0:
                        select.append(item[0] + ", " + item[1])
                        exist_movies.add(num)
            np.random.shuffle(select)  # 乱序
            insert["select"] = select

            train_prompts.append(insert)
            if index % 1000 == 0:
                print(index)

        print("test len:", len(train_prompts))
        with open("test.json", "w") as f:
            json.dump(train_prompts, f)

    def generate_validation_prompt(self, select_num=4):

        train_prompts = []
        for index, row in self.validation_set.iterrows():
            user = users[row["userID"]]
            insert = {}

            gender = user[2]
            age = user[1]
            occupation = user[3]

            if gender == "female":
                prompt = f"""Given a user who is '{gender}, {age} years old, and occupation is {occupation}', her movie viewing history over time includes: """
            else:
                prompt = f"""Given a user who is '{gender}, {age} years old, and occupation is {occupation}', his movie viewing history over time includes: """

            for i, v in enumerate(row["history"]):
                item = items[v]
                # 存在无打分情况？
                if len(row["history_feedback"]) > i:
                    item_scale = row["history_feedback"][i]
                else:
                    item_scale = 3
                prompt += f"({item[0]}, {item[1]}, {item_scale} star); "
            prompt = prompt[:-2] + "."
            insert["prompt"] = prompt

            value = items[row["itemID"]]

            cot_prompt = f"""Please briefly describe the user's viewing preferences and briefly analyze the reasons why the user decides to watch {value[0]} ({value[1]}) next. (within 200 words)"""

            insert["cot_prompt"] = cot_prompt
            insert["value"] = value[0] + ", " + value[1]
            # insert["value_scale"] = row["rating"]

            # 生成随机选项
            exist_movies = row["history"]
            exist_movies.append(row["itemID"])
            exist_movies = set(exist_movies)
            select = [value[0] + ", " + value[1]]
            while len(select) < select_num:
                num = random.randint(1, self.n_items)
                if num not in exist_movies:
                    item = items[num]
                    if len(item) > 0:
                        select.append(item[0] + ", " + item[1])
                        exist_movies.add(num)
            np.random.shuffle(select)  # 乱序
            insert["select"] = select

            train_prompts.append(insert)
            if index % 1000 == 0:
                print(index)

        print("validation len:", len(train_prompts))
        with open("validation.json", "w") as f:
            json.dump(train_prompts, f)


ml_data = Dataset(data)
ml_data.process_data()
# ml_data.generate_train_prompt()
# ml_data.generate_test_prompt()
# ml_data.generate_validation_prompt()
