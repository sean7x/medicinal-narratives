import pandas as pd

# Import and initialize tqdm for Pandas
from tqdm import tqdm
tqdm.pandas()

from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from transformers import BertTokenizer, BertModel
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np

# Depress DeprecationWarnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os.path
from sklearn.model_selection import KFold


def feature_engineering(procd_data, ngram, bert, bert_pretrained_model=None, tfidf=True, RANDOM_SEED=42):
    # BERT Embeddings
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
    
    # Extract BOW and TF-IDF features
    if tfidf:
        # Add bigrams
        if ngram == 'bigram':
            phrase_model = Phrases(procd_data, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)
            procd_data_bigram = procd_data.progress_apply(lambda x: phrase_model[x])
    
            dictionary = Dictionary(procd_data_bigram)
        elif ngram == 'unigram':
            dictionary = Dictionary(procd_data)
    
        # BoW
        bow_corpus = [dictionary.doc2bow(doc) for doc in tqdm(procd_data, desc='Generating BoW')]
    
        # TF-IDF
        tfidf_model = TfidfModel(bow_corpus)
        tfidf_corpus = [tfidf_model[doc] for doc in tqdm(bow_corpus, desc='Generating TF-IDF')]
    
    return len(procd_data), bert_embeddings if bert else None, procd_data_bigram if ngram == 'bigram' else None, dictionary if tfidf else None, bow_corpus if tfidf else None, tfidf_corpus if tfidf else None


def clustering(bert_embeddings, algorithm, num_clusters, RANDOM_SEED):
    """
    Cluster the input data using the specified algorithm and number of clusters.
    """
    # Prepare the input data for clustering
    bert_embedding_avg = [np.mean(embedding, axis=0) for embedding in bert_embeddings]
    input_data = np.vstack(bert_embedding_avg)

    # Scale the input data
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)

    # Initialize clustering algorithm
    if algorithm == 'kmeans':
        clustering_model = KMeans(
            n_clusters=num_clusters,
            n_init='auto',
            random_state=RANDOM_SEED,
        )
    elif algorithm == 'hierarchical':
        clustering_model = AgglomerativeClustering(
            n_clusters=num_clusters,
        )
    else:
        raise ValueError(f'Unknown clustering algorithm: {algorithm}')

    # Fit the clustering algorithm to the data and get the labels
    clustering_model.fit(input_data)
    labels = clustering_model.labels_

    # Calculate the metrics if possible
    if len(set(labels)) > 1:
        silhouette = silhouette_score(input_data, labels)
        davies_bouldin = davies_bouldin_score(input_data, labels)
        calinski_harabasz = calinski_harabasz_score(input_data, labels)
    else:
        silhouette_avg = davies_bouldin = calinski_harabasz = np.nan

    return clustering_model, silhouette, davies_bouldin, calinski_harabasz


if __name__ == '__main__':
    from dvclive import Live
    import pickle
    import gc

    # Path and settings
    RANDOM_SEED = 42
    train_data_path = './data/raw/lewtun-drug-reviews/train.jsonl'

    procd_texts = ['lemma_wo_stpwrd', 'lemma_w_stpwrd']
    bert_pretrained_model = 'bert-base-uncased'

    clustering_algorithms = ['kmeans', 'hierarchical']
    nums_clusters = [2, 3, 4, 5]

    # Load the processed training data
    procd_df_path = './cross_validation/procd_train.csv'

    if os.path.isfile(procd_df_path):
        procd_df = pd.read_csv(procd_df_path)
        #procd_df = procd_df.sample(frac=0.001, random_state=RANDOM_SEED)
    else:
        print('Processed training data not found. Please run `src/preprocessing.py` first.')
    
    # Clustering cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    with Live(dir='./cross_validation/cluster', report='html') as live:
        for fold, (train, test) in tqdm(enumerate(kf.split(procd_df)), desc=f'Folds'):
            print(f'Fold {fold}')
            train_procd_df = procd_df.iloc[train]
            test_procd_df = procd_df.iloc[test]

            for procd_text in tqdm(procd_texts, desc=f'Preprocessed Text'):
                print(procd_text)
                # Load preprocessed text data
                train_procd_data = train_procd_df[procd_text].progress_apply(lambda x: eval(x))
                train_procd_data = train_procd_data[train_procd_data.progress_apply(lambda row: len(row) > 0)]

                test_procd_data = test_procd_df[procd_text].progress_apply(lambda x: eval(x))
                test_procd_data = test_procd_data[test_procd_data.progress_apply(lambda row: len(row) > 0)]

                # Extract bert embeddings from the training data if not already extracted
                if not os.path.isfile(f'./cross_validation/bert/train_bert_embeddings_{fold}_{procd_text}.pkl'):
                    _, train_bert_embeddings, _, _, _, _ = feature_engineering(
                        procd_data = train_procd_data,
                        ngram = None,
                        bert = True,
                        bert_pretrained_model = bert_pretrained_model,
                        tfidf = False,
                        RANDOM_SEED = RANDOM_SEED
                    )
                    with open(f'./cross_validation/bert/train_bert_embeddings_{fold}_{procd_text}.pkl', 'wb') as f:
                        pickle.dump(train_bert_embeddings, f)
                else:
                    with open(f'./cross_validation/bert/train_bert_embeddings_{fold}_{procd_text}.pkl', 'rb') as f:
                        train_bert_embeddings = pickle.load(f)
                # Free up memory
                del train_procd_data
                del train_bert_embeddings
                gc.collect()
                
                # Extract bert embeddings from the test data if not already extracted
                if not os.path.isfile(f'./cross_validation/bert/test_bert_embeddings_{fold}_{procd_text}.pkl'):
                    _, test_bert_embeddings, _, _, _, _ = feature_engineering(
                        procd_data = test_procd_data,
                        ngram = None,
                        bert = True,
                        bert_pretrained_model = bert_pretrained_model,
                        tfidf = False,
                        RANDOM_SEED = RANDOM_SEED
                    )
                    with open(f'./cross_validation/bert/test_bert_embeddings_{fold}_{procd_text}.pkl', 'wb') as f:
                        pickle.dump(test_bert_embeddings, f)
                else:
                    with open(f'./cross_validation/bert/test_bert_embeddings_{fold}_{procd_text}.pkl', 'rb') as f:
                        test_bert_embeddings = pickle.load(f)
                # Free up memory
                del test_procd_data
                gc.collect()

                # Prepare the input test data for clustering
                test_bert_embedding_avg = [np.mean(embedding, axis=0) for embedding in test_bert_embeddings]
                test_data = np.vstack(test_bert_embedding_avg)
                # Scale the test data
                scaler = StandardScaler()
                test_data = scaler.fit_transform(test_data)

                # Load the train bert embeddings
                with open(f'./cross_validation/bert/train_bert_embeddings_{fold}_{procd_text}.pkl', 'rb') as f:
                    train_bert_embeddings = pickle.load(f)
                
                # Train the clustering models
                for clustering_algorithm in tqdm(clustering_algorithms, desc=f'Training clustering models'):
                    print(f'{clustering_algorithm}')
                    for num_clusters in tqdm(nums_clusters):
                        print(f'{num_clusters} clusters')
                        
                        clustering_model, silhouette, davies_bouldin, calinski_harabasz = clustering(
                            bert_embeddings = train_bert_embeddings,
                            algorithm = clustering_algorithm,
                            num_clusters = num_clusters,
                            RANDOM_SEED = RANDOM_SEED
                        )
        
                        live.log_param('Preprocessed Text', procd_text)
                        live.log_param('Clustering Algorithm', clustering_algorithm)
                        live.log_param('Num of Clusters', num_clusters)
                            
                        live.log_metric('Fold', fold)
                        live.log_metric(
                            f'Silhouette - {procd_text} - {clustering_algorithm} - {num_clusters} - Train',
                            silhouette
                        )
                        live.log_metric(
                            f'Davies_Bouldin - {procd_text} - {clustering_algorithm} - {num_clusters} - Train',
                            davies_bouldin
                        )
                        live.log_metric(
                            f'Calinski_Harabasz - {procd_text} - {clustering_algorithm} - {num_clusters} - Train',
                            calinski_harabasz
                        )
        
                        # Evaluate the model
                        # Predict the clustering labels for the test data
                        if clustering_algorithm == 'kmeans':
                            labels = clustering_model.predict(test_data)
                        elif clustering_algorithm == 'hierarchical':
                            labels = clustering_model.fit_predict(test_data)
                        # Calculate the metrics if possible
                        if len(set(labels)) > 1:
                            silhouette = silhouette_score(test_data, labels)
                            davies_bouldin = davies_bouldin_score(test_data, labels)
                            calinski_harabasz = calinski_harabasz_score(test_data, labels)
                        else:
                            silhouette_avg = davies_bouldin = calinski_harabasz = np.nan
                        
                        live.log_metric(
                            f'Silhouette - {procd_text} - {clustering_algorithm} - {num_clusters} - Test',
                            silhouette
                        )
                        live.log_metric(
                            f'Davies_Bouldin - {procd_text} - {clustering_algorithm} - {num_clusters} - Test',
                            davies_bouldin
                        )
                        live.log_metric(
                            f'Calinski_Harabasz - {procd_text} - {clustering_algorithm} - {num_clusters} - Test',
                            calinski_harabasz
                        )

            live.next_step()