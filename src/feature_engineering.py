from dvclive import Live
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, Word2Vec
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import torch
import pickle


def main(args):
    with Live() as live:
        procd_data = pd.read_csv(args.input_path)['procd_review'].apply(lambda x: x.split())
        
        # Initialize
        if args.bow or args.tfidf:
            dictionary = Dictionary(procd_data)
        
        if args.word2vec:
            word2vec_model = Word2Vec(
                vector_size=args.vector_size,
                window=args.window,
                min_count=args.min_count,
                workers=args.workers,
                epochs=args.epochs,
                sg=0,
                seed=42
            )
        
        if args.bert:
            torch.manual_seed(42)
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # BoW
        if args.bow:
            bow_corpus = [dictionary.doc2bow(doc) for doc in tqdm(procd_data, desc='Generating BoW')]

            with open('data/features/bow_corpus.pkl', 'wb') as f:
                pickle.dump(bow_corpus, f)
        
        # TF-IDF
        if args.tfidf:
            tfidf_model = TfidfModel(bow_corpus)
            tfidf_corpus = [tfidf_model[doc] for doc in tqdm(bow_corpus, desc='Generating TF-IDF')]

            with open('data/features/tfidf_corpus.pkl', 'wb') as f:
                pickle.dump(tfidf_corpus, f)
        
        # Word2Vec
        if args.word2vec:
            word2vec_model.build_vocab(tqdm(procd_data, desc='Building Word2Vec Vocab'))
            word2vec_model.train(
                tqdm(procd_data, desc='Training Word2Vec'),
                total_examples=word2vec_model.corpus_count,
                epochs=word2vec_model.epochs
            )

            with open('data/features/word2vec_model.pkl', 'wb') as f:
                pickle.dump(word2vec_model, f)
        
        # BERT Embeddings
        if args.bert:
            def get_bert_embeddings(text):
                inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                outputs = bert_model(**inputs)
                return outputs.last_hidden_state.mean(dim=1)
            
            bert_embeddings = [
                get_bert_embeddings(doc).detach().numpy() for doc in tqdm(procd_data, desc='Generating BERT Embeddings')
            ]

            with open('data/features/bert_embeddings.pkl', 'wb') as f:
                pickle.dump(bert_embeddings, f)
        
        # Logging the metrics with DVCLive
        live.log_metric('Number of Documents', len(procd_data))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Feature Engineering Parameters')
    parser.add_argument(
        '--input_path', type=str, required=True,
        help='Path to the input preprocessed data file'
    )
    parser.add_argument(
        '--bow', type=bool, default=True,
        help='Use Bag-of-Words'
    )
    parser.add_argument(
        '--tfidf', type=bool, default=True,
        help='Use TF-IDF'
    )
    parser.add_argument(
        '--word2vec', type=bool, default=True,
        help='Use Word2Vec'
    )
    parser.add_argument(
        '--vector_size', type=int, default=100,
        help='Vector size for Word2Vec'
    )
    parser.add_argument(
        '--window', type=int, default=5,
        help='Window size for Word2Vec'
    )
    parser.add_argument(
        '--min_count', type=int, default=1,
        help='Minimum count for Word2Vec'
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Number of epochs for Word2Vec'
    )
    parser.add_argument(
        '--workers', type=int, default=-1,
        help='Number of workers for Word2Vec'
    )
    parser.add_argument(
        '--bert', type=bool, default=True,
        help='Use BERT Embeddings'
    )

    args = parser.parse_args()
    main(args)