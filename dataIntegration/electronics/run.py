import json
from prompt import test_openai_api


def generate_cot(filename, demoname=None):

    prefixs = ""

    if demoname:
        with open(demoname, "r") as f:
            data = json.loads(f)
            for i in data:
                prefixs += "viewing history: " + i["history"] + "\n"
                prefixs += "Please analyze the user's preferences in 100 words based on the viewing history.\n"
                prefixs += i["preferences"] + "\n"

    fp = None
    with open(filename, "r") as f:
        fp = json.load(f)

    for i, v in enumerate(fp[:]):
        if v.get("preference"):
            continue

        prom = (
            "A user's purchase record over time includes: "
            + v["history"]
            + " Please summary the user's preferences within 60 words."
        )
        fp[i]["preference"] = test_openai_api(prefixs + prom)

        print(filename, "已生成数量：", i)
        if i % 10 == 0 or i == len(fp) - 1:
            # print("运行时间：", end - start)
            with open(filename, "w") as wf:
                json.dump(fp, wf)


# generate_cot("train.json")
# generate_cot("val.json")
# generate_cot("test.json")
