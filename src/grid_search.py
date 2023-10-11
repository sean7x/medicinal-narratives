import itertools
import subprocess


# Grid searches
def cluster_grid_search():
    # Grid search hyperparameters for clustering
    num_clusterses = [5, 10, 15, 20, 25, 30]
    epses = [0.5, 1, 2, 3, 4, 5]
    min_sampleses = [2, 3, 4, 5, 6, 7]

    print("Clustering hyperparameter search: \n")

    for num_clusters, eps, min_samples in itertools.product(num_clusterses, epses, min_sampleses):
        subprocess.run([
            "dvc", "exp", "run", "--queue",
            "--set-param", "clustering_bert.num_clusters={num_clusters}",
            "--set-param", "clustering_bert.eps={eps}",
            "--set-param", "clustering_bert.min_samples={min_samples}",
            "-m", f"Clustering with num_clusters: {num_clusters}, eps: {eps}, min_samples: {min_samples}",
        ])
        print(f"""num_clusters: {num_clusters}, eps: {eps}, min_samples: {min_samples} \n
              has been added to the DVC queue. \n""")
    
    print("Run `dvc exp run --run-all` to start the grid search.")


def topic_grid_search():
    # Grid search hyperparameters for topic modeling
    nums_topics = [10, 20, 30, 40, 50]

    print("Topic modeling hyperparameter search: \n")

    for num_topics in nums_topics:
        subprocess.run([
            "dvc", "exp", "run", "--queue",
            "--set-param", "topic_modeling_bow_tfidf.num_topics={num_topics}",
            "-m", f"Topic modeling with num_topics: {num_topics}",
        ])
        print(f"""num_topics: {num_topics} \n
              has been added to the DVC queue. \n""")
    
    print("Run `dvc exp run --run-all` to start the grid search.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grid search")
    parser.add_argument("--cluster", action="store_true", help="Clustering grid search")
    parser.add_argument("--topic", action="store_true", help="Topic modeling grid search")
    args = parser.parse_args()

    if args.cluster:
        cluster_grid_search()
    elif args.topic:
        topic_grid_search()
    else:
        print("Please specify a grid search to run.")