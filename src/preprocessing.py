import spacy
import pandas as pd
import html

# Import and initialize tqdm for Pandas
from tqdm import tqdm
tqdm.pandas()


def preprocess(input_path, output_path):
    nlp = spacy.load('en_core_web_sm')

    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix == '.jsonl':
        df = pd.read_json(input_path, lines=True)
    elif input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    elif input_path.suffix == '.feather':
        df = pd.read_feather(input_path)
    else:
        raise ValueError(f"Unknown file type: {input_path.suffix}")
    
    # Decode HTML entities back to original characters and remove whitespaces
    df['review'] = df['review'].apply(html.unescape).str.replace(r'[\r\n\t]', '', regex=True).str.strip()

    # Remove wrong condition values and keep the rows
    df.loc[df.condition.notna() & df.condition.str.contains('users found this comment helpful'), 'condition'] = None
    
    # Remove rows with empty reviews
    df = df[df['review'].notna()]
    df = df[df['review'] != '"-"']
    df = df[df['review'] != '']

    # Generate lemmas for each token, remove stopwords and punctuations
    #df['procd_review'] = df['review'].progress_apply(
    #    #lambda x: ' '.join([token.lemma_ for token in nlp(x) if not token.is_stop and not token.is_punct])
    #    lambda x: [token.lemma_ for token in nlp(x) if not token.is_stop and not token.is_punct]
    #)
    def lemma(row):
        # Skip if review is empty
        if pd.isnull(row['review']): return row

        # lemma_w_stpwrd: with stop words, for word2vec and bert embeddings
        row['lemma_w_stpwrd'] = [token.lemma_ for token in nlp(row['review']) if not (token.is_punct or token.is_space or token.lemma_.strip() == '')]
        # lemma_wo_stpwrd: lower without stop words, for BoW and TF-IDF embeddings
        row['lemma_wo_stpwrd'] = [token.lemma_.lower() for token in nlp(row['review']) if not (token.is_stop or token.is_punct or token.is_space or token.lemma_.strip() == '')]
        
        # For reviews with only stop words, use lemma_w_stpwrd
        if len(row['lemma_wo_stpwrd']) == 0:
            row['lemma_wo_stpwrd'] = row['lemma_w_stpwrd']
        return row
    
    df = df.progress_apply(lemma, axis=1)

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Preprocess dataset.')
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to the input dataset"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to save the output preprocessed dataset"
    )

    args = parser.parse_args()

    preprocess(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path)
    )