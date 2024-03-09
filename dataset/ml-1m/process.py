import pandas as pd
import copy
import json
import random
import numpy as np


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
        threshold=4,
        order=True,
        leave_n=1,
        keep_n=5,
        max_history_length=5,
        premise_threshold=0,
    ):
        # filter ratings by threshold
        self.proc_dataset = self.dataset.copy()
        # self.proc_dataset["rating"][self.proc_dataset["rating"] < threshold] = 0
        # self.proc_dataset["rating"][self.proc_dataset["rating"] >= threshold] = 1
        if order:
            self.proc_dataset = self.proc_dataset.sort_values(
                by=["timestamp", "userID", "itemID"]
            ).reset_index(drop=True)

        self.leave_one_out_by_time(leave_n, keep_n)
        self.generate_histories(max_hist_length=max_history_length, premise_threshold=0)

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

    def generate_prompt(self, select_num=4):

        train_prompts = []
        for index, row in self.train_set.iterrows():
            user = self.user_dataset.loc[
                self.user_dataset["userID"] == row["userID"]
            ].values[0]
            insert = {}

            gender = user[1]
            age = user[2]
            occupation = user[3]

            if gender == "female":
                prompt = f"""Given a user who is '{gender}, {age} years old, and occupation is {occupation}', her movie viewing history over time includes: """
            else:
                prompt = f"""Given a user who is '{gender}, {age} years old, and occupation is {occupation}', his movie viewing history over time includes: """

            for i, v in enumerate(row["history"]):
                item = self.item_dataset.loc[self.item_dataset["itemID"] == v].values[0]
                item_scale = row["history_feedback"][i]
                prompt += f"({item[1]}, {item[2]}, {item_scale} star); "
            prompt = prompt[:-2] + "."
            insert["prompt"] = prompt

            value = self.item_dataset.loc[
                self.item_dataset["itemID"] == row["itemID"]
            ].values[0]

            if gender == "female":
                cot_prompt = f"""She will watch {value[1]} ({value[2]}) next, please provide clear explanations based on details from the user's viewing history and other pertinent factors."""
            else:
                cot_prompt = f"""He will watch {value[1]} ({value[2]}) next, please provide clear explanations based on details from the user's viewing history and other pertinent factors."""

            insert["cot_prompt"] = cot_prompt
            insert["value"] = value[1] + ", " + value[2]
            # insert["value_scale"] = row["rating"]

            # 生成随机选项
            exist_movies = row["history"]
            exist_movies.append(row["itemID"])
            exist_movies = set(exist_movies)
            select = [value[1] + ", " + value[2]]
            while len(select) < select_num:
                num = random.randint(0, self.n_items)
                if num not in exist_movies:
                    item = self.item_dataset.loc[
                        self.item_dataset["itemID"] == num
                    ].values
                    if len(item) > 0:
                        item = item[0]
                        select.append(item[1] + ", " + item[2])
                        exist_movies.add(num)
            np.random.shuffle(select)  # 乱序
            insert["select"] = select

            train_prompts.append(insert)
            if index % 1000 == 0:
                print(index)
        with open("train.json", "w") as f:
            json.dump(train_prompts, f)

    def generate_test_prompt(self, select_num=4):

        train_prompts = []
        for index, row in self.test_set.iterrows():
            user = self.user_dataset.loc[
                self.user_dataset["userID"] == row["userID"]
            ].values[0]
            insert = {}

            gender = user[1]
            age = user[2]
            occupation = user[3]

            if gender == "female":
                prompt = f"""Given a user who is '{gender}, {age} years old, and occupation is {occupation}', her movie viewing history over time includes: """
            else:
                prompt = f"""Given a user who is '{gender}, {age} years old, and occupation is {occupation}', his movie viewing history over time includes: """

            for i, v in enumerate(row["history"]):
                item = self.item_dataset.loc[self.item_dataset["itemID"] == v].values[0]
                item_scale = row["history_feedback"][i]
                prompt += f"({item[1]}, {item[2]}, {item_scale} star); "
            prompt = prompt[:-2] + "."
            insert["prompt"] = prompt

            value = self.item_dataset.loc[
                self.item_dataset["itemID"] == row["itemID"]
            ].values[0]

            if gender == "female":
                cot_prompt = f"""She will watch {value[1]} ({value[2]}) next, please provide clear explanations based on details from the user's viewing history and other pertinent factors."""
            else:
                cot_prompt = f"""He will watch {value[1]} ({value[2]}) next, please provide clear explanations based on details from the user's viewing history and other pertinent factors."""

            insert["cot_prompt"] = cot_prompt
            insert["value"] = value[1] + ", " + value[2]
            # insert["value_scale"] = row["rating"]

            # 生成随机选项
            exist_movies = row["history"]
            exist_movies.append(row["itemID"])
            exist_movies = set(exist_movies)
            select = [value[1] + ", " + value[2]]
            while len(select) < select_num:
                num = random.randint(0, self.n_items)
                if num not in exist_movies:
                    item = self.item_dataset.loc[
                        self.item_dataset["itemID"] == num
                    ].values
                    if len(item) > 0:
                        item = item[0]
                        select.append(item[1] + ", " + item[2])
                        exist_movies.add(num)
            np.random.shuffle(select)
            insert["select"] = select

            train_prompts.append(insert)
            if index % 1000 == 0:
                print(index)
        with open("test.json", "w") as f:
            json.dump(train_prompts, f)

    def generate_validation_prompt(self, select_num=4):

        train_prompts = []
        for index, row in self.validation_set.iterrows():
            user = self.user_dataset.loc[
                self.user_dataset["userID"] == row["userID"]
            ].values[0]
            insert = {}

            gender = user[1]
            age = user[2]
            occupation = user[3]

            if gender == "female":
                prompt = f"""Given a user who is '{gender}, {age} years old, and occupation is {occupation}', her movie viewing history over time includes: """
            else:
                prompt = f"""Given a user who is '{gender}, {age} years old, and occupation is {occupation}', his movie viewing history over time includes: """

            for i, v in enumerate(row["history"]):
                item = self.item_dataset.loc[self.item_dataset["itemID"] == v].values[0]
                item_scale = row["history_feedback"][i]
                prompt += f"({item[1]}, {item[2]}, {item_scale} star); "
            prompt = prompt[:-2] + "."
            insert["prompt"] = prompt

            value = self.item_dataset.loc[
                self.item_dataset["itemID"] == row["itemID"]
            ].values[0]

            if gender == "female":
                cot_prompt = f"""She will watch {value[1]} ({value[2]}) next, please provide clear explanations based on details from the user's viewing history and other pertinent factors."""
            else:
                cot_prompt = f"""He will watch {value[1]} ({value[2]}) next, please provide clear explanations based on details from the user's viewing history and other pertinent factors."""

            insert["cot_prompt"] = cot_prompt
            insert["value"] = value[1] + ", " + value[2]
            # insert["value_scale"] = row["rating"]

            # 生成随机选项
            exist_movies = row["history"]
            exist_movies.append(row["itemID"])
            exist_movies = set(exist_movies)
            select = [value[1] + ", " + value[2]]
            while len(select) < select_num:
                num = random.randint(0, self.n_items)
                if num not in exist_movies:
                    item = self.item_dataset.loc[
                        self.item_dataset["itemID"] == num
                    ].values
                    if len(item) > 0:
                        item = item[0]
                        select.append(item[1] + ", " + item[2])
                        exist_movies.add(num)
            np.random.shuffle(select)
            insert["select"] = select

            train_prompts.append(insert)
            if index % 1000 == 0:
                print(index)
        with open("validation.json", "w") as f:
            json.dump(train_prompts, f)


if __name__ == "__main__":

    file = "ratings.dat"
    rating_data = get_ratings(file)
    print(rating_data)

    file = "movies.dat"
    item_data = get_items(file)
    print(item_data)

    file = "users.dat"
    user_data = get_users(file)
    print(user_data)

    data = Dataset(rating_data, user_data, item_data)
    data.process_data()
    data.generate_prompt()
    # data.generate_test_prompt()
    # data.generate_validation_prompt()
