import spacy
import pandas as pd


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
    
    # Generate lemmas for each token, remove stopwords and punctuations, and join back into a string
    df['procd_review'] = df['review'].apply(
        lambda x: ' '.join([token.lemma_ for token in nlp(x) if not token.is_stop and not token.is_punct])
    )

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