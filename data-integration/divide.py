import json


def divide():
    with open("train.json", "r") as rf:
        data = json.load(rf)
        length = len(data)
        print("length: ", length)
        with open("train-1.json", "w") as wf:
            json.dump(data[: length // 6], wf)
        with open("train-2.json", "w") as wf:
            json.dump(data[length // 6 : length // 6 * 2], wf)
        with open("train-3.json", "w") as wf:
            json.dump(data[length // 6 * 2 : length // 6 * 3], wf)
        with open("train-4.json", "w") as wf:
            json.dump(data[length // 6 * 3 : length // 6 * 4], wf)
        with open("train-5.json", "w") as wf:
            json.dump(data[length // 6 * 4 : length // 6 * 5], wf)
        with open("train-6.json", "w") as wf:
            json.dump(data[length // 6 * 5 :], wf)


def merge():
    data = []
    with open("train-1.json", "r") as rf:
        data += json.load(rf)
    with open("train-2.json", "r") as rf:
        data += json.load(rf)
    with open("train-3.json", "r") as rf:
        data += json.load(rf)
    with open("train-4.json", "r") as rf:
        data += json.load(rf)
    with open("train-5.json", "r") as rf:
        data += json.load(rf)
    with open("train-6.json", "r") as rf:
        data += json.load(rf)
    with open("train.json", "w") as wf:
        json.dump(data, wf)


# divide()
merge()
