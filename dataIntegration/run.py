import argparse
from ml100k.process1 import Dataset as ml100kDataset
from ml100k.process2 import add_detail as ml100kAddDetail
from ml1m.process1 import Dataset as ml1mDataset
from ml1m.process2 import add_detail as ml1mAddDetail
from electronics.process1 import Dataset as electronicsDataset
from electronics.process2 import add_detail as electronicsAddDetail
from movies.process1 import Dataset as moviesDataset
from movies.process2 import add_detail as moviesAddDetail
from cluster_process import main as cluster_main
from generate import generate_cot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k")
    parser.add_argument("--cot", type=str, default="zero-shot")

    args = parser.parse_args()
    return args


def main(args):
    # Template Construction
    if args.dataset == "ml100k":
        ml100kDataset()
        ml100kAddDetail("train.json")
        ml100kAddDetail("val.json")
        ml100kAddDetail("test.json")
    elif args.dataset == "ml1m":
        ml1mDataset()
        ml1mAddDetail("train.json")
        ml1mAddDetail("val.json")
        ml1mAddDetail("test.json")
    elif args.dataset == "electronics":
        electronicsDataset()
        electronicsAddDetail("train.json")
        electronicsAddDetail("val.json")
        electronicsAddDetail("test.json")
    elif args.dataset == "movies":
        moviesDataset()
        moviesAddDetail("train.json")
        moviesAddDetail("val.json")
        moviesAddDetail("test.json")

    # CoT Construction
    if args.cot == "cluster":
        cluster_main()
        generate_cot("train.json", "demo/cluster_prompt.json")
        generate_cot("val.json", "demo/cluster_prompt.json")
        generate_cot("test.json", "demo/cluster_prompt.json")
    elif args.cot == "manual":
        generate_cot("train.json", "demo/manual_prompt.json")
        generate_cot("val.json", "demo/manual_prompt.json")
        generate_cot("test.json", "demo/manual_prompt.json")
    else:
        generate_cot("train.json")
        generate_cot("val.json")
        generate_cot("test.json")


if __name__ == "__main__":

    args = parse_args()
    print("args", args)
    print("====Input Arguments====")
    main(args=args)
