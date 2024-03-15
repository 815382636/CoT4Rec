import argparse
import numpy as np
import torch
import random
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def parse_arguments():
    parser = argparse.ArgumentParser(description="CoT-Rec")
    parser.add_argument(
        "--task",
        type=str,
        default="ml-1m",
        choices=["ml-1m"],
        help="dataset used for experiment",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        default="dataset/ml-1m",
    )
    parser.add_argument(
        "--max_ra_len", type=int, default=5, help="maximum number of prompt examples"
    )
    parser.add_argument(
        "--demo_save_dir",
        type=str,
        default="demos/ml-1m",
        help="where to save the contructed demonstrations",
    )
    parser.add_argument("--num_clusters", type=int, default=8, help="cluster count")
    parser.add_argument("--random_seed", type=int, default=192, help="random seed")
    parser.add_argument(
        "--encoder",
        type=str,
        default="all-MiniLM-L6-v2",
        help="which sentence-transformer encoder for clustering",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="center",
        help="whether to sample the cluster center first",
    )
    args = parser.parse_args()
    return args


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_arguments()
    fix_seed(args.random_seed)

    task = args.task
    pred_file = args.pred_file
    save_file = args.demo_save_dir
    max_ra_len = args.max_ra_len
    num_clusters = args.num_clusters

    encoder = SentenceTransformer(args.encoder)
    corpus = []
    questions = []
    with open(pred_file + "/test.json", "r") as f:
        fp = json.load(f)
        for i in fp:
            corpus.append(i["prompt"] + i["cot_prompt"])
            questions.append(i["prompt"] + i["cot_prompt"])

    print("corpus len:", len(corpus))

    corpus_embeddings = encoder.encode(corpus)

    # Perform kmean clustering
    clustering_model = KMeans(n_clusters=num_clusters, random_state=args.random_seed)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for _ in range(num_clusters)]

    dist = clustering_model.transform(corpus_embeddings)
    clustered_dists = [[] for _ in range(num_clusters)]
    clustered_idx = [[] for _ in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])
        clustered_dists[cluster_id].append(dist[sentence_id][cluster_id])
        clustered_idx[cluster_id].append(sentence_id)

    demos = []

    for i in range(len(clustered_dists)):
        print("Cluster ", i + 1)
        tmp = list(map(list, zip(range(len(clustered_dists[i])), clustered_dists[i])))
        top_min_dist = sorted(tmp, key=lambda x: x[1], reverse=False)

        for element in top_min_dist:
            min_idx = element[0]
            c_question = questions[clustered_idx[i][min_idx]]
            demos.append(c_question)

    print(demos)
    demos = {"demo": demos}

    with open(args.demo_save_dir, "w", encoding="utf-8") as write_f:
        json.dump(demos, write_f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
