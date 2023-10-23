if __name__ == '__main__':
    from gensim.models import LdaModel, nmf
    from gensim.corpora import Dictionary
    import pandas as pd
    import dvc.api
    import pickle

    # Load parameters from DVC
    params = dvc.api.params_show()

    # Load the topic model
    if params['topic_modeling']['algorithm'] == 'lda':
        topic_model = LdaModel.load(params['topic_model_path'])
    elif params['topic_modeling']['algorithm'] == 'nmf':
        topic_model = nmf.Nmf.load(params['topic_model_path'])