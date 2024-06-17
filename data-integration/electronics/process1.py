import json
import pandas as pd
import re


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


meta_data = {}  # id2name
with open("meta_Electronics.json", "r") as rf:
    for i in rf.readlines():
        data = i
        data = data[data.find("'asin': '") + 9 :]
        asin = data[: data.find("', ")]
        # 存在大量无title问题
        if data.find("'title': ") == -1:
            continue
        else:
            data = data[data.find("'title': ") + 10 :]
            if data.find(", '") != -1:
                title = data[: data.find(", '") - 1]
            else:
                title = data[:-2]
        print(asin, "\t", title)
        meta_data[asin] = title
        # break

raw_dataset = pd.read_csv("ratings_Electronics.csv")


class Dataset(object):
    def __init__(self, dataset, u2i=True):
        self.dataset = dataset
        self.use_u2i = u2i
        self.n_users = self.dataset["userID"].nunique()
        self.n_items = self.dataset["itemID"].nunique()
        self.meta_data_new = set()
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
        self.proc_dataset = self.dataset.copy()
        if order:
            self.proc_dataset = self.proc_dataset.sort_values(
                by=["timestamp", "userID", "itemID"]
            ).reset_index(drop=True)
        self.generate_data_train(history_length=max_history_length, target_length=5)

    def generate_data_train(self, history_length, target_length, step=3):
        train_set = []
        test_set = []
        validation_set = []

        processed_data = self.proc_dataset.copy()
        # 数量 29241
        rrr = []
        for uid, group in processed_data.groupby("userID"):  # group by uid
            user_list = []

            # 判断是否存在无名称问题
            new_list = []
            for test in group.index:
                if meta_data.get(group.loc[test, "itemID"]):
                    new_list.append(test)

            ind = len(new_list) - history_length - target_length
            rrr.append(len(new_list))
            while ind >= 0:
                if ind + 30 <= len(new_list):
                    user_list.append(list(new_list[ind : ind + 30]))
                else:
                    if len(new_list) < 30:
                        user_list.append(list(new_list[ind:]) + list(new_list[:ind]))
                    else:
                        user_list.append(
                            list(new_list[ind:])
                            + list(new_list[: 30 - (len(new_list) - ind)])
                        )
                ind -= step
            if len(user_list) > 3:
                # 划分train、test、val
                start = """"""
                prompt = start
                result = []
                front = []
                for index, i in enumerate(user_list[-1]):
                    m_name = deal_with(meta_data[group.loc[i, "itemID"]])
                    self.meta_data_new.add(m_name)
                    if index > 9:
                        result.append(m_name)
                    else:
                        front.append(m_name)
                        v1 = m_name
                        v2 = int(group.loc[i, "rating"])
                        prompt += f"({v1}, {v2} star); "
                prompt = prompt[:-2] + "."
                validation_set.append(
                    {"history": prompt, "result": result, "front": front}
                )

                prompt = start
                result = []
                front = []
                for index, i in enumerate(user_list[-2]):
                    m_name = deal_with(meta_data[group.loc[i, "itemID"]])
                    self.meta_data_new.add(m_name)
                    if index > 9:
                        result.append(m_name)
                    else:
                        front.append(m_name)
                        v1 = m_name
                        v2 = int(group.loc[i, "rating"])
                        prompt += f"({v1}, {v2} star); "
                prompt = prompt[:-2] + "."
                test_set.append({"history": prompt, "result": result, "front": front})

                for freq in user_list[:-2]:
                    prompt = start
                    result = []
                    front = []
                    for index, i in enumerate(freq):
                        m_name = deal_with(meta_data[group.loc[i, "itemID"]])
                        self.meta_data_new.add(m_name)
                        if index > 9:
                            result.append(m_name)
                        else:
                            front.append(m_name)
                            v1 = m_name
                            v2 = int(group.loc[i, "rating"])
                            prompt += f"({v1}, {v2} star); "
                    prompt = prompt[:-2] + "."
                    train_set.append(
                        {"history": prompt, "result": result, "front": front}
                    )

        with open("name.txt", "w") as f:
            for i in self.meta_data_new:
                f.write(i + "\n")

        print("min(rrr)", min(rrr))
        print("max(rrr)", max(rrr))
        print("meta_data_new.len", len(self.meta_data_new))
        print("meta_data.len", len(meta_data))

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
    data = Dataset(raw_dataset)
    data.process_data()
