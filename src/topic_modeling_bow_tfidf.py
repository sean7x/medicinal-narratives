import pickle
from gensim.models import LdaModel, Nmf, CoherenceModel
from gensim.corpora import Dictionary
import pandas as pd
from dvclive import Live


def main(args):
    # Load data
    #procd_data = pd.read_csv(args.procd_data_path)['procd_review']

    with open(args.bow_corpus_path, 'rb') as f:
        bow_corpus = pickle.load(f)
    
    with open(args.tfidf_corpus_path, 'rb') as f:
        tfidf_corpus = pickle.load(f)
    
    # Load dictionary
    dictionary = Dictionary.load(args.dictionary_path)


    with Live() as live:
        # LDA Model for BoW
        lda_bow = LdaModel(
            bow_corpus,
            num_topics=args.lda_n_topics,
            id2word=dictionary,
            random_state=42,
        )

        # Calculate Coherence and Perplexity for LDA with BoW
        coherence_model_lda_bow = CoherenceModel(
            model=lda_bow,
            #texts=procd_data.tolist(),
            corpus=bow_corpus,
            dictionary=dictionary,
            #coherence='c_v',
            coherence='u_mass',
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
            num_topics=args.lda_n_topics,
            id2word=dictionary,
            random_state=42,
        )

        # Calculate Coherence and Perplexity for LDA with TF-IDF
        coherence_model_lda_tfidf = CoherenceModel(
            model=lda_tfidf,
            #texts=procd_data.tolist(),
            corpus=tfidf_corpus,
            dictionary=dictionary,
            #coherence='c_v',
            coherence='u_mass',
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
            num_topics=args.nmf_n_topics,
            id2word=dictionary,
            random_state=42,
        )

        # Calculate Coherence for NMF with BoW
        coherence_model_nmf_bow = CoherenceModel(
            model=nmf_bow,
            #texts=procd_data.tolist(),
            corpus=bow_corpus,
            dictionary=dictionary,
            #coherence='c_v',
            coherence='u_mass',
            processes=-1
        )
        coherence_nmf_bow = coherence_model_nmf_bow.get_coherence()

        # Log metrics for NMF with BoW
        live.log_metric('Coherence (NMF, BoW)', coherence_nmf_bow)


        # NMF Model for TF-IDF
        nmf_tfidf = Nmf(
            tfidf_corpus,
            num_topics=args.nmf_n_topics,
            id2word=dictionary,
            random_state=42,
        )

        # Calculate Coherence for NMF with TF-IDF
        coherence_model_nmf_tfidf = CoherenceModel(
            model=nmf_tfidf,
            #texts=procd_data.tolist(),
            corpus=tfidf_corpus,
            dictionary=dictionary,
            #coherence='c_v',
            coherence='u_mass',
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

        live.next_step()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--procd_data_path', type=str, required=True)
    parser.add_argument('--bow_corpus_path', type=str, required=True)
    parser.add_argument('--tfidf_corpus_path', type=str, required=True)
    parser.add_argument('--dictionary_path', type=str, required=True)
    parser.add_argument('--lda_n_topics', type=int, required=True)
    parser.add_argument('--nmf_n_topics', type=int, required=True)
    parser.add_argument('--lda_bow_model_path', type=str, required=True)
    parser.add_argument('--lda_tfidf_model_path', type=str, required=True)
    parser.add_argument('--nmf_bow_model_path', type=str, required=True)
    parser.add_argument('--nmf_tfidf_model_path', type=str, required=True)
    args = parser.parse_args()

    main(args)