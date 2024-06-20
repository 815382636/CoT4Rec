import os
import numpy as np
import torch
import json
import argparse
import random
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
)
from utils_data import load_dataset_std, DatasetStd
from rich.console import Console
from torch.utils.data import DataLoader
from metrics4rec import evaluate_all
from tqdm import tqdm


console = Console(record=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-100k")
    parser.add_argument("--output_dir", type=str, default="experiments")
    parser.add_argument("--input_len", type=int, default=1024)
    parser.add_argument("--output_len", type=int, default=128)
    parser.add_argument("--eval_bs", type=int, default=16)
    parser.add_argument(
        "--use_generate",
        action="store_true",
        help="only for baseline to improve inference speed",
    )
    parser.add_argument(
        "--evaluate_dir",
        type=str,
        default=None,
        help="the directory of model for evaluation",
    )
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="REC-PA",
        help="prompt format template",
        choices=["REC-P", "REC-PA", "REC-A", "REC-LLM-PA"],
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--stage", type=int, default=2, help="one or two stages")

    args = parser.parse_args()
    return args


def T5Test(args):

    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")

    save_dir = args.evaluate_dir
    print("save_dir:", save_dir)

    _, _, test_data = load_dataset_std(args)

    model = T5ForConditionalGeneration.from_pretrained(args.model)

    test_set = DatasetStd(
        test_data,
        tokenizer,
        args.input_len,
        args.output_len,
        args.stage,
        args,
    )

    test_loader = DataLoader(
        dataset=test_set,
        shuffle=False,
        num_workers=0,
        batch_size=args.eval_bs,
        pin_memory=True,
    )

    model.cuda()
    all_info = []
    for i, v in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            beam_outputs = model.generate(
                v["input_ids"].to("cuda"),
                max_length=50,
                num_beams=20,
                no_repeat_ngram_size=0,
                num_return_sequences=20,
                early_stopping=True,
            )
            generated_sents = tokenizer.batch_decode(
                beam_outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            targets = test_data[i * args.eval_bs : (i + 1) * args.eval_bs]
            for j in range(len(targets)):
                new_info = {}
                new_info["target_item"] = targets[j]["result"][0]
                new_info["gen_item_list"] = generated_sents[j * 20 : (j + 1) * 20]
                all_info.append(new_info)

    gt = {}
    ui_scores = {}
    for i, info in enumerate(all_info):
        gt[i] = [info["target_item"]]
        pred_dict = {}
        for j in range(len(info["gen_item_list"])):
            try:
                pred_dict[info["gen_item_list"][j]] = -(j + 1)
            except:
                pass
        ui_scores[i] = pred_dict

    evaluate_all(ui_scores, gt, 5)
    evaluate_all(ui_scores, gt, 10)


if __name__ == "__main__":

    args = parse_args()
    print("args", args)
    print("====Input Arguments====")
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)  # 设置随机数种子，保证随机数一样

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    T5Test(args=args)
