if __name__ == '__main__':
    import argparse
    import pandas as pd
    import dvc.api
    import pickle
    from gensim.models import LdaModel, Nmf, CoherenceModel
    from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
    from gensim.corpora import Dictionary
    from gensim.models import TfidfModel
    from dvclive import Live
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--topic_model_path', type=str, required=True)
    parser.add_argument('--dictionary_path', type=str, required=True)

    args = parser.parse_args()

    # Load parameters from DVC
    params = dvc.api.params_show()
    
    with Live("evaluate_topic_model") as live:
        # Load the topic model
        if params['topic_modeling']['algorithm'] == 'lda':
            topic_model = LdaModel.load(args.topic_model_path)
        elif params['topic_modeling']['algorithm'] == 'nmf':
            topic_model = Nmf.load(args.topic_model_path)
        
        # Load the dictionary
        dictionary = Dictionary.load(args.dictionary_path)
        
        # Set the paths to the preprocessed data
        ngram = params['feature_engineering']['ngram']
        if ngram == 'unigram':
            procd_data_paths = {
            'train': f"data/preprocessed/procd_{params['train_data_path']}.csv",
            'test': f"data/preprocessed/procd_{params['test_data_path']}.csv"
        }
        elif params['feature_engineering']['ngram'] == 'bigram':
            procd_data_paths = {
            'train': f"data/preprocessed/procd_{params['train_data_path']}_{params['procd_text']}_bigram.csv",
            'test': f"data/preprocessed/procd_{params['test_data_path']}.csv"
        }

        for type in tqdm(['train', 'test']):
            # Load the preprocessed data
            procd_data_path = procd_data_paths[type]
            procd_df = pd.read_csv(procd_data_path)
            procd_data = procd_df[params['procd_text']].apply(lambda x: eval(x))
            procd_data = procd_data[procd_data.apply(lambda row: len(row) > 0)]

            # Add bigrams to the preprocessed test data
            if (type == 'test') and (ngram == 'bigram'):
                phrase_model = Phrases(procd_data, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)
                procd_data = procd_data.apply(lambda x: phrase_model[x])
            
            # Calculate the coherence score
            coherence_model = CoherenceModel(
                model=topic_model,
                texts=procd_data.tolist(),
                dictionary=dictionary,
                coherence='c_v',
                processes=-1
            )
            coherence = coherence_model.get_coherence()
            live.log_metric(f"Coherence - {type}", coherence)

            # Extract the topics distribution and the dominant topic
            if type == 'train':
                if not params['feature_engineering']['tfidf']:
                    corpus = pickle.load(open(f"data/features/{params['procd_text']}/bow_{ngram}_corpus.pkl", 'rb'))
                else:
                    corpus = pickle.load(open(f"data/features/{params['procd_text']}/tfidf_{ngram}_corpus.pkl", 'rb'))
            else:
                corpus = [dictionary.doc2bow(doc) for doc in tqdm(procd_data, desc='Generating BoW corpus for test data')]
                if params['feature_engineering']['tfidf']:
                    tfidf_model = TfidfModel(corpus)
                    corpus = [tfidf_model[doc] for doc in tqdm(corpus, desc='Generating TF-IDF corpus for test data')]
            
            #topics_dist = [topic_model.get_document_topics(doc) for doc in tqdm(corpus, desc='Extracting topics distribution for test data')]
            topics_dist = pd.DataFrame([
                {topic: prop for topic, prop in topic_model.get_document_topics(doc)}
                for doc in tqdm(corpus, desc='Extracting topics distribution for test data')
            ])
            topics_dist.columns = [f"Topic {i}" for i in range(topic_model.num_topics)]
            topics_dist.fillna(0, inplace=True) # Fill the NaN values with 0
            topics_dist.to_csv(f"data/evaluate/topics_dist_{type}.csv", index=False)

            #dominant_topic = topics_dist.apply(lambda row: row.idxmax(), axis=1)
            dominant_topic = topics_dist.idxmax(axis=1)
            dominant_topic.to_csv(f"data/evaluate/dominant_topic_{type}.csv", index=False)

            # Extract the topic keywords
            topic_keywords = pd.DataFrame(topic_model.get_topics())
            topic_keywords.columns = dictionary.values()
            topic_keywords.to_csv(f"data/evaluate/topic_keywords_{type}.csv", index=False)

            # Extract the topic coherence
            topic_coherence = pd.DataFrame([coherence_model.get_coherence_per_topic()])
            topic_coherence.columns = [f"Topic {i}" for i in range(topic_model.num_topics)]
            topic_coherence.to_csv(f"data/evaluate/topic_coherence_{type}.csv", index=False)

            live.next_step()