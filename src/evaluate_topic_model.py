if __name__ == '__main__':
    import argparse
    import pandas as pd
    import dvc.api
    import pickle
    from gensim.models import LdaModel, Nmf, CoherenceModel
    from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
    from gensim.corpora import Dictionary
    from dvclive import Live

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

        for type in ['train', 'test']:
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
            live.next_step()