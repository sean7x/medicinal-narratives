from sklearn.model_selection import train_test_split
import pandas as pd


def extract_stratified_sample(input_path, output_path, fraction, stratify_column):
    df = pd.read_csv(input_path)
    _, sample_df = train_test_split(df, test_size=fraction, stratify=df[stratify_column])

    sample_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Extract a stratified sample from dataset.')
    parser.add_argument(
        "input_path", type=str, required=True,
        help="Path to the input dataset"
    )
    parser.add_argument(
        "output_path", type=str, required=True,
        help="Path to save the output sample"
    )
    parser.add_argument(
        "--fraction", type=float, default=0.2,
        help="Fraction of data to sample"
    )
    parser.add_argument(
        '--stratify_column', type=str, required=True,
        help='Column to use for stratification'
    )

    args = parser.parse_args()

    extract_stratified_sample(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        fraction=args.fraction,
        stratify_column=args.stratify_column
    )