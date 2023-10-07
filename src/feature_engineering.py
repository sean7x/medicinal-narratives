from dvclive import Live
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, Word2Vec
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import torch
import pickle


def main(input_path, bow, tfidf, word2vec, vector_size, window, min_count, epochs, sg, RANDOM_SEED, bert):
    with Live() as live:
        #procd_data = pd.read_csv(args.input_path)['procd_review'].apply(lambda x: x.split())
        procd_data = pd.read_csv(args.input_path)['procd_review'].apply(lambda x: eval(x))
        
        # Initialize
        if bow or tfidf:
            dictionary = Dictionary(procd_data)
            dictionary.save('data/features/dictionary.pkl')
        
        if word2vec:
            word2vec_model = Word2Vec(
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=-1,
                epochs=epochs,
                sg=sg,
                seed=RANDOM_SEED
            )
        
        if bert:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertModel.from_pretrained('bert-base-uncased')
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
        
        # Word2Vec
        if word2vec:
            word2vec_model.build_vocab(tqdm(procd_data, desc='Building Word2Vec Vocab'))
            word2vec_model.train(
                tqdm(procd_data, desc='Training Word2Vec'),
                total_examples=word2vec_model.corpus_count,
                epochs=word2vec_model.epochs
            )

            with open('data/features/word2vec_model.pkl', 'wb') as f:
                pickle.dump(word2vec_model, f)
        
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
        '--input_path', type=str, required=True,
        help='Path to the input preprocessed data file'
    )
    args = parser.parse_args()

    params = dvc.api.params_show()

    main(
        input_path = Path(args.input_path),
        bow = params['feature_engineering']['bow'],
        tfidf = params['feature_engineering']['tfidf'],
        word2vec = params['feature_engineering']['word2vec'],
        vector_size = params['feature_engineering']['word2vec']['vector_size'],
        window = params['feature_engineering']['word2vec']['window'],
        min_count = params['feature_engineering']['word2vec']['min_count'],
        epochs = params['feature_engineering']['word2vec']['epochs'],
        sg = params['feature_engineering']['word2vec']['sg'],
        RANDOM_SEED = params['RANDOM_SEED'],
        bert = params['feature_engineering']['bert']
    )