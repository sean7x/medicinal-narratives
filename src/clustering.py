from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from gensim.models import LdaModel, Nmf
import numpy as np
from tqdm import tqdm


def cluster(input_data, algorithm, RANDOM_SEED, kwargs):
    """
    Cluster the input data using the specified algorithm and number of clusters.
    """
    # Scale the input data
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)

    # Initialize clustering algorithm
    if algorithm == 'kmeans':
        model = KMeans(
            n_clusters=kwargs['num_clusters'],
            n_init='auto',
            random_state=RANDOM_SEED,
        )
    elif algorithm == 'dbscan':
        model = DBSCAN(
            eps=kwargs['eps'],
            min_samples=kwargs['min_samples'],
            n_jobs=-1,
        )
    elif algorithm == 'hierarchical':
        model = AgglomerativeClustering(
            n_clusters=kwargs['num_clusters'],
            #distance_threshold=kwargs['distance_threshold'],
        )
    else:
        raise ValueError(f'Unknown clustering algorithm: {algorithm}')

    # Fit the clustering algorithm to the data and get the labels
    model.fit(input_data)
    labels = model.labels_

    # Calculate the metrics if possible
    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(input_data, labels)
        davies_bouldin = davies_bouldin_score(input_data, labels)
        calinski_harabasz = calinski_harabasz_score(input_data, labels)
    else:
        silhouette_avg = davies_bouldin = calinski_harabasz = np.nan

    return model, silhouette_avg, davies_bouldin, calinski_harabasz


def prepare_input_data(model, model_type, corpus=None):
    """
    Prepare the input data for clustering.
    """
    if model_type == 'lda':
        input_data = np.array(
            [model.get_document_topics(doc, minimum_probability=0) for doc in corpus]
        )[:, :, 1]
        return input_data
    
    elif model_type == 'nmf':
        topic_term_matrix = model.get_topics()
        num_topics = model.num_topics
        input_data = []

        for doc in tqdm(model[corpus]):
            doc_topics = dict(doc)
            doc_topic_vec = [doc_topics.get(i, 0.0) for i in range(num_topics)]
            input_data.append(doc_topic_vec)
        
        return np.array(input_data)
    
    elif model_type == 'word2vec':
        input_data = model.wv.vectors
        return input_data
    
    elif model_type == 'bert':
        bert_embedding_avg = [np.mean(embedding, axis=0) for embedding in model]
        input_data = np.vstack(bert_embedding_avg)
        return input_data


if __name__ == '__main__':
    import argparse
    import pickle
    from dvclive import Live
    import dvc.api

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_paths', type=str, required=True)
    argparser.add_argument('--corpus_paths', type=str, required=False)
    argparser.add_argument('--log_dir', type=str, required=False)
    args = argparser.parse_args()

    model_paths = args.model_paths.split(',')
    if args.corpus_paths is not None: corpus_paths = args.corpus_paths.split(',')

    params = dvc.api.params_show()
    RANDOM_SEED = params['RANDOM_SEED']
    kwargs = params['clustering']

    with Live(dir=args.log_dir, report="html") as live:
        for model_path in model_paths:
            model_name = model_path.split('/')[-1].split('_model')[0].split('_embeddings')[0]
            #live.step = model_name

            if 'lda' in model_path:
                if 'bow' in model_path:
                    corpus_path = [corpus for corpus in corpus_paths if 'bow' in corpus][0]
                elif 'tfidf' in model_path:
                    corpus_path = [corpus for corpus in corpus_paths if 'tfidf' in corpus][0]
                model = LdaModel.load(model_path)
                corpus = pickle.load(open(corpus_path, 'rb'))
                input_data = prepare_input_data(model, 'lda', corpus)
            elif 'nmf' in model_path:
                if 'bow' in model_path: 
                    corpus_path = [corpus for corpus in corpus_paths if 'bow' in corpus][0]
                elif 'tfidf' in model_path: 
                    corpus_path = [corpus for corpus in corpus_paths if 'tfidf' in corpus][0]
                model = Nmf.load(model_path)
                corpus = pickle.load(open(corpus_path, 'rb'))
                input_data = prepare_input_data(model, 'nmf', corpus)
            elif 'word2vec' in model_path:
                model = pickle.load(open(model_path, 'rb'))
                input_data = prepare_input_data(model, 'word2vec')
            elif 'bert' in model_path:
                model = pickle.load(open(model_path, 'rb'))
                input_data = prepare_input_data(model, 'bert')
            
            for algorithm in ['kmeans', 'dbscan', 'hierarchical']:
                model, silhouette_avg, davies_bouldin, calinski_harabasz = cluster(
                    input_data,
                    algorithm,
                    RANDOM_SEED,
                    kwargs,
                )

                live.step = f'{model_name}_{algorithm}'

                live.log_metric(f'Silhouette Score', silhouette_avg)
                live.log_metric(f'Davies-Bouldin Score', davies_bouldin)
                live.log_metric(f'Calinski-Harabasz Score', calinski_harabasz)

                pickle.dump(model, open(f'./models/{model_name}_{algorithm}.pkl', 'wb'))
                
        live.make_summary()
        live.make_report()
        live.make_dvcyaml()
