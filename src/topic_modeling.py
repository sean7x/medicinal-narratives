import pickle
from gensim.models import LdaModel, Nmf, CoherenceModel
from gensim.corpora import Dictionary
import pandas as pd
from tqdm import tqdm
import pyLDAvis
import numpy as np

# Depress DeprecationWarnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def prepare_topic_model_viz(topic_model, dictionary, corpus):
    # Extract the topic-term matrix
    topic_term_matrix = topic_model.get_topics()

    # Extract the document-topic matrix
    num_topics = topic_model.num_topics
    doc_topic_matrix = []

    for doc in tqdm(topic_model[corpus]):
        doc_topics = dict(doc)
        doc_topic_vec = [doc_topics.get(i, 0.0) for i in range(num_topics)]
        doc_topic_matrix.append(doc_topic_vec)

    # Normalize topic_term_matrix and doc_topic_matrix
    topic_term_matrix = topic_term_matrix / np.sum(topic_term_matrix, axis=1, keepdims=True)
    doc_topic_matrix = doc_topic_matrix / np.sum(doc_topic_matrix, axis=1, keepdims=True)

    doc_topic_matrix = np.array(doc_topic_matrix)
    
    # Vocabulary and term frequencies
    vocab = [dictionary[i] for i in range(len(dictionary))]

    term_freq = np.zeros(len(vocab))
    for doc in corpus:
        for idx, freq in doc:
            term_freq[idx] += freq
    
    # Prepare the data in pyLDAvis format
    vis_data = pyLDAvis.prepare(
        doc_lengths=np.array([sum(dict(doc).values()) for doc in corpus]),
        vocab=vocab,
        term_frequency=term_freq,
        topic_term_dists=topic_term_matrix,
        doc_topic_dists=doc_topic_matrix
    )

    # Return the visualization data
    return vis_data


def topic_modeling(procd_data, corpus, dictionary, kwargs, RANDOM_SEED):
    # Set up topic model
    # LDA Model
    if kwargs['algorithm'] == 'lda':
        topic_model = LdaModel(
            corpus,
            num_topics=kwargs['num_topics'],
            id2word=dictionary,
            random_state=RANDOM_SEED,
        )
        perplexity = topic_model.log_perplexity(corpus)
    
    # NMF Model
    elif kwargs['algorithm'] == 'nmf':
        topic_model = Nmf(
            corpus,
            num_topics=kwargs['num_topics'],
            id2word=dictionary,
            random_state=RANDOM_SEED,
        )
        perplexity = None

    # Calculate Coherence score
    coherence_model = CoherenceModel(
        model=topic_model,
        texts=procd_data.tolist(),
        #corpus=corpus,
        dictionary=dictionary,
        coherence='c_v',
        #coherence='u_mass',
        processes=-1
    )
    coherence = coherence_model.get_coherence()

    return topic_model, perplexity, coherence


if __name__ == '__main__':
    import argparse
    import dvc.api
    from dvclive import Live
    from pathlib import Path
    from collections import defaultdict

    parser = argparse.ArgumentParser()
    parser.add_argument('--procd_data_path', type=str, required=True)
    parser.add_argument('--dictionary_path', type=str, required=True)
    args = parser.parse_args()

    params = dvc.api.params_show()
    kwargs = params['topic_modeling']

    with Live(dir="topic_modeling", resume=True, report="html") as live:
        # Load preprocessed text data
        procd_data = pd.read_csv(Path(args.procd_data_path))[kwargs['procd_text']].apply(lambda x: eval(x))

        if kwargs['feature'] == 'bow':
            corpus = pickle.load(open(Path('./data/features/bow_corpus.pkl'), 'rb'))
        elif kwargs['feature'] == 'tfidf':
            corpus = pickle.load(open(Path('./data/features/tfidf_corpus.pkl'), 'rb'))
        
        # Load dictionary
        dictionary = Dictionary.load(args.dictionary_path)

        # Add cluster labels to corpus
        if not kwargs['cluster_model'] == None:
            # Load clustering model
            cluster_model = pickle.load(open(Path(kwargs['cluster_model']), 'rb'))
            # Get the all the cluster labels
            labels = cluster_model.labels_

            # Add cluster labels to the preprocessed text data
            grouped_procd_data = pd.DataFrame({'cluster_label': labels, 'procd_text': procd_data}).groupby('cluster_label')

            # Apply LDA to each clustered corpus
            topic_models = {}
            coherence_scores = {}
            perplexity_scores = {}

            for label, group in grouped_procd_data:
                # Extract the clustered corpus and texts
                clustered_corpus = [corpus[i] for i in group.index]
                clustered_texts = group['procd_text']
                
                # Train the topic model for this cluster
                #topic_model = LdaModel(corpus=clustered_corpus, id2word=dictionary, num_topics=10, random_state=42)  # Adjust num_topics as needed
                topic_model, perplexity, coherence = topic_modeling(clustered_texts, clustered_corpus, dictionary, kwargs, params['RANDOM_SEED'])

                # Save models
                topic_model.save(f"./models/{kwargs['algorithm']}_{kwargs['feature']}_{label}_model.pkl")
                
                # Save the topic model
                topic_models[label] = topic_model
                
                # Save the scores
                coherence_scores[label] = coherence
                if perplexity is not None: perplexity_scores[label] = perplexity

            coherence = np.mean(list(coherence_scores.values()))
            if len(perplexity_scores) == 0: perplexity = None
            else: perplexity = np.mean(list(perplexity_scores.values()))
        
        else:
            topic_model, perplexity, coherence = topic_modeling(procd_data, corpus, dictionary, kwargs, params['RANDOM_SEED'])

            # Save models
            topic_model.save(f"./models/{kwargs['algorithm']}_{kwargs['feature']}_model.pkl")

            # Prepare visualization data for topic_model and save the visualization
            pyLDAvis.save_html(
                prepare_topic_model_viz(topic_model, dictionary, corpus),
                f"./models/{kwargs['algorithm']}_{kwargs['feature']}_vis.pkl"
            )
        
        live.log_metric('Coherence', coherence)
        if perplexity is not None: live.log_metric('Perplexity', perplexity)