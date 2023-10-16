from dvclive import Live
import pandas as pd
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, Word2Vec
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import torch
import pickle


def main(procd_data_path, bow, tfidf, bigram, bert, bert_pretrained_model, RANDOM_SEED):
    with Live(dir='feature_engineering') as live:
        # Load preprocessed text data
        procd_data = pd.read_csv(Path(procd_data_path))[params['procd_text']].apply(lambda x: eval(x))
        procd_data = procd_data[procd_data.apply(lambda row: len(row) > 0)]

        # Initialize
        if bow or tfidf:
            # Add bigrams
            if bigram:
                phrase_model = Phrases(procd_data, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)
                procd_data = procd_data.apply(lambda x: phrase_model[x])
                procd_data.to_csv('{}_bigram.csv'.format(procd_data_path.split('.csv')[0]), index=False)

            dictionary = Dictionary(procd_data)
            dictionary.save('data/features/dictionary.pkl')
        
        if bert:
            tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model)
            bert_model = BertModel.from_pretrained(bert_pretrained_model)
            # Move model to GPU if available
            if torch.cuda.is_available():
                device = torch.device('cuda')
                torch.cuda.manual_seed_all(RANDOM_SEED)
                torch.backends.cudnn.deterministic = True
            #elif torch.backends.mps.is_available():
            #    device = torch.device('mps')
            #    torch.mps.manual_seed(RANDOM_SEED)
            #    torch.backends.mps.deterministic = True
            else:
                device = torch.device('cpu')
                torch.manual_seed(RANDOM_SEED)
                torch.backends.cudnn.deterministic = True

            bert_model = bert_model.to(device)
            print(f"Using device: {device}")
        
        # BoW
        if bow:
            bow_corpus = [dictionary.doc2bow(doc) for doc in tqdm(procd_data, desc='Generating BoW')]

            with open('data/features/bow_corpus.pkl', 'wb') as f:
                pickle.dump(bow_corpus, f)
        
        # TF-IDF
        if tfidf:
            tfidf_model = TfidfModel(bow_corpus)
            tfidf_corpus = [tfidf_model[doc] for doc in tqdm(bow_corpus, desc='Generating TF-IDF')]

            with open('data/features/tfidf_corpus.pkl', 'wb') as f:
                pickle.dump(tfidf_corpus, f)
        
        # BERT Embeddings
        if bert:
            def get_bert_embeddings(text):
                inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                # Move inputs to GPU if available
                if device.type != 'cpu':
                    inputs = {key: val.to(device) for key, val in inputs.items()}

                outputs = bert_model(**inputs)
                return outputs.last_hidden_state.mean(dim=1)
            
            bert_embeddings = [
                get_bert_embeddings(doc).cpu().detach().numpy() for doc in tqdm(procd_data, desc='Generating BERT Embeddings')
            ]

            with open('data/features/bert_embeddings.pkl', 'wb') as f:
                pickle.dump(bert_embeddings, f)
        
        # Logging the metrics with DVCLive
        live.log_metric('Number of Documents', len(procd_data))


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    import dvc.api

    parser = argparse.ArgumentParser(description='Feature Engineering Parameters')
    parser.add_argument(
        '--procd_data_path', type=str, required=True,
        help='Path to the input preprocessed data file'
    )
    args = parser.parse_args()

    params = dvc.api.params_show()
    kwargs = params['feature_engineering']

    main(
        procd_data_path = args.procd_data_path,
        bow = kwargs['bow'],
        tfidf = kwargs['tfidf'],
        bigram = kwargs['bigram'],
        bert = kwargs['bert'],
        bert_pretrained_model = kwargs['bert_pretrained_model'],
        RANDOM_SEED = params['RANDOM_SEED']
    )