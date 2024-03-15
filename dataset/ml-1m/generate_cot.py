import json
import time
from scope import call_with_messages

fp = None
with open("test_cot.json", "r") as f:
    fp = json.load(f)

start = time.perf_counter()

for i, v in enumerate(fp[:]):
    if v.get("rational"):
        continue

    fp[i]["rational"] = call_with_messages(v["prompt"] + " " + v["cot_prompt"])

    print("已生成数量：", i)
    if i % 10 == 0:
        end = time.perf_counter()
        print("运行时间：", end - start)
        with open("test_cot.json", "w") as wf:
            json.dump(fp, wf)
