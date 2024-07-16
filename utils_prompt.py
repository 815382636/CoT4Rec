def construct_source(history, preference=None, rec_list=None, dataset="ml-100k"):
    if dataset == "electronics":
        if not preference:
            # 没有使用用户偏好
            input = f"The user's previous purchase record and their ratings were: {history} "
            if rec_list:
                input += "\nThe products available for selection include: "
                r_list = "; ".join(rec_list)
                input += r_list + " ."
            input += "\nBased on his purchase record, please recommend a movie for him."
        else:
            input = f"The user's previous purchase record and their ratings were: {history} \n His Personal preferences: {preference}"
            if rec_list:
                input += "\nThe products available for selection include: "
                r_list = " ; ".join(rec_list)
                input += r_list + " ."
            input += "\nBased on his purchase record and preferences, please recommend a product for him."
    else:

        if not preference:
            # 没有使用用户偏好
            input = f"The user's previous order of watching movies and their ratings were: {history} "
            if rec_list:
                input += "\nThe movies available for selection include: "
                r_list = "; ".join(rec_list)
                input += r_list + " ."
            input += "\nBased on his viewing history, please recommend a movie for him."
        else:
            input = f"The user's previous order of watching movies and their ratings were: {history} \n His Personal preferences: {preference}"
            if rec_list:
                input += "\nThe movies available for selection include: "
                r_list = " ; ".join(rec_list)
                input += r_list + " ."
            input += "\nBased on his viewing history and preferences, please recommend a movie for him."

    return input


def construct_pre_source(history, dataset="ml-100k"):
    if dataset == "electronics":
        return f"The user's previous purchase record and their ratings were: {history} Please analyze his preference characteristics based on the previous behavior."
    else:
        return f"The user's previous order of watching movies and their ratings were: {history} Please analyze his preference characteristics based on the previous behavior."

    # return f"Viewing history (From Ancient Times to the Present): {history}.\nPlease analyze my personal preferences in a short paragraph based on the viewing history."


def construct_target(target_list):
    return "; ".join(target_list)


def build(data, form, stage, dataset):
    # 请根据我的观影历史和个人喜好，为我定制并排序一个电影推荐列表。
    # Based on my viewing history and personal preferences, please customize and sort a curated list of movie recommendations for me.
    # 基于我的观影历史和个人偏好，请您为我定制并排序一份精选电影推荐清单。可选的电影包括：
    # Based on my viewing history and personal preferences, please customize and sort a curated list of movie recommendations for me. The movies available for selection include:

    #  choices=["REC-P", "REC-PA", "REC-A"],
    if "A" in form:
        if form == "REC-PA":
            prompt = (
                construct_source(
                    data["history"],
                    data["preference"],
                    data["recommendations"],
                    dataset,
                )
                if stage == 1
                else construct_source(
                    data["history"],
                    data["pred_preference"],
                    data["recommendations"],
                    dataset,
                )
            )
        elif form == "REC-A":
            prompt = construct_source(
                data["history"], None, data["recommendations"], dataset
            )
        # target = construct_target(data["result"][:10])
        target = data["result"][0]
    # 训练用户偏好
    else:
        prompt = construct_pre_source(data["history"], dataset)
        target = data["preference"]

    return prompt, target
