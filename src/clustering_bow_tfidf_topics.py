import pickle
from gensim.models import LdaModel, Nmf
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from dvclive import Live


def load_gensim_model(model_path):
    return LdaModel.load(model_path) if 'lda' in model_path else Nmf.load(model_path)


def get_topic_word_distribution(model, num_topics):
    topic_word_distributions = []
    for i in range(num_topics):
        topic_terms = model.get_topic_terms(i)
        word_probs = np.zeros(model.num_terms)
        for word_id, prob in topic_terms:
            word_probs[word_id] = prob
        topic_word_distributions.append(word_probs)
    return np.array(topic_word_distributions)


def cluster_and_evaluate(X, algorithm, **kwargs):
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=kwargs.get('n_clusters', 5), random_state=42)
    elif algorithm == 'dbscan':
        model = DBSCAN()