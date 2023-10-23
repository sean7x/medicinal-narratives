if __name__ == '__main__':
    from sklearn.decomposition import PCA
    import numpy as np
    import pickle
    import pandas as pd

    with open('data/features/bert_embeddings.pkl', 'rb') as f:
        bert_embeddings = pickle.load(f)
    
    n_components = 50
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(np.vstack(bert_embeddings))
    #reduced_embeddings = pca.fit_transform(bert_embeddings)

    print("Explained Variance Ratio: {}".format(pca.explained_variance_ratio_))
    print("Total Variance Explained: {}".format(np.sum(pca.explained_variance_ratio_)))

    df = pd.DataFrame(reduced_embeddings)
    df.to_parquet('data/features/bert_embeddings_pca.parquet', index=False)