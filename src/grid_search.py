import itertools
import subprocess


# Grid searches
def feature_grid_search():
    # Grid search parameters for word2vec
    vector_sizes = [100, 200, 300]
    window_sizes = [5, 10, 15]
    min_word_counts = [1, 5, 10]
    epochs = [5, 10, 15]
    sgs = [0, 1]

    print("Feature engineering parameter search: \n")

    for vector_size, window, min_count, epoch, sg in itertools.product(vector_sizes, window_sizes, min_word_counts, epochs, sgs):
        subprocess.run([
            "dvc", "exp", "run", "--queue",
            "--set-param", "feature_engineering.vector_size={vector_size}",
            "--set-param", "feature_engineering.window={window}",
            "--set-param", "feature_engineering.min_count={min_count}",
            "--set-param", "feature_engineering.epoch={epoch}",
            "--set-param", "feature_engineering.sg={sg}",
        ])
        print(f"""vector_size: {vector_size}, window: {window}, min_count: {min_count}, epoch: {epoch}, sg: {sg} \n
              has been added to the DVC queue. \n""")
    
    print("Run `dvc exp run --run-all` to start the grid search.")


def topic_grid_search():
    # Grid search parameters topic modeling
    num_topics = [10, 20, 30, 40, 50]

    print("Topic modeling parameter search: \n")

    for num_topic in num_topics:
        subprocess.run([
            "dvc", "exp", "run", "--queue",
            "--set-param", "topic_modeling_bow_tfidf.lda_num_topics={num_topic}",
            "--set-param", "topic_modeling_bow_tfidf.nmf_num_topics={num_topic}",
        ])
        print(f"""lda_num_topics: {num_topic}, nmf_num_topics: {num_topic} \n
              has been added to the DVC queue. \n""")
    
    print("Run `dvc exp run --run-all` to start the grid search.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grid search")
    parser.add_argument("--feature", action="store_true", help="Feature engineering grid search")
    parser.add_argument("--topic", action="store_true", help="Topic modeling grid search")
    args = parser.parse_args()

    if args.feature:
        feature_grid_search()
    elif args.topic:
        topic_grid_search()
    else:
        print("Please specify a grid search to run.")