import pickle
from gensim.models import LdaModel, Nmf, CoherenceModel
from gensim.corpora import Dictionary
import pandas as pd
from pathlib import Path
from dvclive import Live
from tqdm import tqdm
import pyLDAvis
import numpy as np

# Depress DeprecationWarnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def prepare_topic_model_viz(model, dictionary, corpus):
    # Extract the topic-term matrix
    topic_term_matrix = model.get_topics()

    # Extract the document-topic matrix
    num_topics = model.num_topics
    doc_topic_matrix = []

    for doc in tqdm(model[corpus]):
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


def main(args, num_topics, RANDOM_SEED):
    # Load data
    #procd_data = pd.read_csv(args.procd_data_path)['procd_review'].apply(lambda x: x.split())
    procd_data = pd.read_csv(Path(args.procd_data_path))['lemma_wo_stpwrd'].apply(lambda x: eval(x))

    with open(Path(args.bow_corpus_path), 'rb') as f:
        bow_corpus = pickle.load(f)
    
    with open(Path(args.tfidf_corpus_path), 'rb') as f:
        tfidf_corpus = pickle.load(f)
    
    # Load dictionary
    dictionary = Dictionary.load(args.dictionary_path)


    with Live(dir="topic_modeling", report="html") as live:
        # LDA Model for BoW
        lda_bow = LdaModel(
            bow_corpus,
            num_topics=num_topics,
            id2word=dictionary,
            random_state=RANDOM_SEED,
        )

        # Calculate Coherence and Perplexity for LDA with BoW
        coherence_model_lda_bow = CoherenceModel(
            model=lda_bow,
            texts=procd_data.tolist(),
            #corpus=bow_corpus,
            dictionary=dictionary,
            coherence='c_v',
            #coherence='u_mass',
            processes=-1
        )
        coherence_lda_bow = coherence_model_lda_bow.get_coherence()
        perplexity_lda_bow = lda_bow.log_perplexity(bow_corpus)

        # Log metrics for LDA with BoW
        live.log_metric('Coherence (LDA, BoW)', coherence_lda_bow)
        live.log_metric('Perplexity (LDA, BoW)', perplexity_lda_bow)

        # LDA Model for TF-IDF
        lda_tfidf = LdaModel(
            tfidf_corpus,
            num_topics=num_topics,
            id2word=dictionary,
            random_state=RANDOM_SEED,
        )

        # Calculate Coherence and Perplexity for LDA with TF-IDF
        coherence_model_lda_tfidf = CoherenceModel(
            model=lda_tfidf,
            texts=procd_data.tolist(),
            #corpus=tfidf_corpus,
            dictionary=dictionary,
            coherence='c_v',
            #coherence='u_mass',
            processes=-1
        )
        coherence_lda_tfidf = coherence_model_lda_tfidf.get_coherence()
        perplexity_lda_tfidf = lda_tfidf.log_perplexity(tfidf_corpus)

        # Log metrics for LDA with TF-IDF
        live.log_metric('Coherence (LDA, TF-IDF)', coherence_lda_tfidf)
        live.log_metric('Perplexity (LDA, TF-IDF)', perplexity_lda_tfidf)


        # NMF Model for BoW
        nmf_bow = Nmf(
            bow_corpus,
            num_topics=num_topics,
            id2word=dictionary,
            random_state=RANDOM_SEED,
        )

        # Calculate Coherence for NMF with BoW
        coherence_model_nmf_bow = CoherenceModel(
            model=nmf_bow,
            texts=procd_data.tolist(),
            #corpus=bow_corpus,
            dictionary=dictionary,
            coherence='c_v',
            #coherence='u_mass',
            processes=-1
        )
        coherence_nmf_bow = coherence_model_nmf_bow.get_coherence()

        # Log metrics for NMF with BoW
        live.log_metric('Coherence (NMF, BoW)', coherence_nmf_bow)


        # NMF Model for TF-IDF
        nmf_tfidf = Nmf(
            tfidf_corpus,
            num_topics=num_topics,
            id2word=dictionary,
            random_state=RANDOM_SEED,
        )

        # Calculate Coherence for NMF with TF-IDF
        coherence_model_nmf_tfidf = CoherenceModel(
            model=nmf_tfidf,
            texts=procd_data.tolist(),
            #corpus=tfidf_corpus,
            dictionary=dictionary,
            coherence='c_v',
            #coherence='u_mass',
            processes=-1
        )
        coherence_nmf_tfidf = coherence_model_nmf_tfidf.get_coherence()

        # Log metrics for NMF with TF-IDF
        live.log_metric('Coherence (NMF, TF-IDF)', coherence_nmf_tfidf)

        # Save models
        lda_bow.save(args.lda_bow_model_path)
        lda_tfidf.save(args.lda_tfidf_model_path)
        nmf_bow.save(args.nmf_bow_model_path)
        nmf_tfidf.save(args.nmf_tfidf_model_path)
 
        # Prepare visualization data for LDA with BoW and save to html
        pyLDAvis.save_html(
            prepare_topic_model_viz(lda_bow, dictionary, bow_corpus),
            args.lda_bow_vis_path
        )
        pyLDAvis.save_html(
            prepare_topic_model_viz(lda_tfidf, dictionary, tfidf_corpus),
            args.lda_tfidf_vis_path
        )
        pyLDAvis.save_html(
            prepare_topic_model_viz(nmf_bow, dictionary, bow_corpus),
            args.nmf_bow_vis_path
        )
        pyLDAvis.save_html(
            prepare_topic_model_viz(nmf_tfidf, dictionary, tfidf_corpus),
            args.nmf_tfidf_vis_path
        )

        #live.next_step()


if __name__ == '__main__':
    import argparse
    import dvc.api

    parser = argparse.ArgumentParser()
    parser.add_argument('--procd_data_path', type=str, required=True)
    parser.add_argument('--bow_corpus_path', type=str, required=True)
    parser.add_argument('--tfidf_corpus_path', type=str, required=True)
    parser.add_argument('--dictionary_path', type=str, required=True)
    parser.add_argument('--lda_bow_model_path', type=str, required=True)
    parser.add_argument('--lda_tfidf_model_path', type=str, required=True)
    parser.add_argument('--nmf_bow_model_path', type=str, required=True)
    parser.add_argument('--nmf_tfidf_model_path', type=str, required=True)
    parser.add_argument('--lda_bow_vis_path', type=str, required=True)
    parser.add_argument('--lda_tfidf_vis_path', type=str, required=True)
    parser.add_argument('--nmf_bow_vis_path', type=str, required=True)
    parser.add_argument('--nmf_tfidf_vis_path', type=str, required=True)
    args = parser.parse_args()

    params = dvc.api.params_show()
    num_topics = params['topic_modeling']['num_topics']

    main(
        args,
        num_topics,
        RANDOM_SEED=params['RANDOM_SEED'],
    )