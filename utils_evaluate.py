"""
Adapted from https://github.com/lupantech/ScienceQA
"""

import os
import json
import argparse
import warnings
import pandas as pd
from sentence_transformers import SentenceTransformer
from evaluations import caculate_bleu, caculate_rouge, caculate_similariry
import math

warnings.filterwarnings("ignore")


def get_acc_with_contion(res_pd, key, values):
    if isinstance(values, list):
        total_pd = res_pd[res_pd[key].isin(values)]
    else:
        total_pd = res_pd[res_pd[key] == values]
    correct_pd = total_pd[total_pd["true_false"] == True]
    acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)
    return acc


def get_scores(rationale_data, results_reference):

    ## BLEU
    bleu1 = caculate_bleu(rationale_data, results_reference, gram=1)
    bleu4 = caculate_bleu(rationale_data, results_reference, gram=4)

    ## Rouge-L
    rouge = caculate_rouge(rationale_data, results_reference)

    ## Similarity
    model = SentenceTransformer("all-MiniLM-L6-v2").cuda()
    similariry = caculate_similariry(rationale_data, results_reference, model)

    scores = {
        "rationale": {
            "bleu1": bleu1 * 100,
            "bleu4": bleu4 * 100,
            "rouge": rouge * 100,
            "similariry": similariry * 100,
        },
    }

    return scores


def print_scores(scores):
    latex_output = ""
    for key, score in scores.items():
        print(f"{key[4:]}: \t{score}")
        latex_output += f"& {score} "
    latex_output += "\\\\"
    print(latex_output)


def caculate(pred, target):
    pred_list = pred.split(";")
    target_list = target.split(";")
    tar = target_list[0]

    hit_5 = 0
    hit_10 = 0
    hit_20 = 0

    ndcg_5 = 0
    ndcg_10 = 0
    ndcg_20 = 0

    mrr_5 = 0
    mrr_10 = 0
    mrr_20 = 0

    count_5 = 0
    count_10 = 0
    count_20 = 0

    if len(target_list) >= 5 and len(pred_list) >= 5:
        count_5 = 1
    if len(target_list) >= 10 and len(pred_list) >= 10:
        count_10 = 1
    if len(target_list) >= 20 and len(pred_list) >= 20:
        count_20 = 1

    if count_5:
        for i in range(5):
            if tar.strip() == pred_list[i].strip():
                hit_5 = 1
                ndcg_5 = 1 / (math.log2((i + 1) + 1))
                mrr_5 = 1 / (i + 1)
    if count_10:
        for i in range(10):
            if tar.strip() == pred_list[i].strip():
                hit_10 = 1
                ndcg_10 = 1 / (math.log2((i + 1) + 1))
                mrr_10 = 1 / (i + 1)
    if count_20:
        for i in range(20):
            if tar.strip() == pred_list[i].strip():
                hit_20 = 1
                ndcg_20 = 1 / (math.log2((i + 1) + 1))
                mrr_20 = 1 / (i + 1)
    return (
        hit_5,
        hit_10,
        hit_20,
        ndcg_5,
        ndcg_10,
        ndcg_20,
        mrr_5,
        mrr_10,
        mrr_20,
        count_5,
        count_10,
        count_20,
    )
