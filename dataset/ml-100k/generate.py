import json
import time
from scope import call_with_messages


def splice_json():

    with open("train.json", "r") as f:
        rf = json.load(f)
        length = len(rf)
        with open("train_1.json", "w") as f1:
            json.dump(rf[: length // 5], f1)
        with open("train_2.json", "w") as f1:
            json.dump(rf[length // 5 : length // 5 * 2], f1)
        with open("train_3.json", "w") as f1:
            json.dump(rf[length // 5 * 2 : length // 5 * 3], f1)
        with open("train_4.json", "w") as f1:
            json.dump(rf[length // 5 * 3 : length // 5 * 4], f1)
        with open("train_5.json", "w") as f1:
            json.dump(rf[length // 5 * 4 :], f1)


# splice_json()


def generate_cot(filename):
    with open(filename, "r") as f:
        fp = json.load(f)

    start = time.perf_counter()

    for i, v in enumerate(fp[:]):
        if v.get("rational"):
            continue

        fp[i]["rational"] = call_with_messages(v["prompt"] + " " + v["cot_prompt"])

        print(filename, "已生成数量：", i)
        if i % 10 == 0:
            end = time.perf_counter()
            # print("运行时间：", end - start)
            with open(filename, "w") as wf:
                json.dump(fp, wf)


# generate_cot("train_1.json")
# generate_cot("train_2.json")
# generate_cot("train_3.json")
# generate_cot("train_4.json")
generate_cot("train_5.json")
