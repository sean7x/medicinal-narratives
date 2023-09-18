import pandas as pd
import random

# Set up the random seed
random.seed(42)

# Read in the data
train_df = pd.read_json('../data/lewtun-drug-reviews/train.jsonl', lines=True)
test_df = pd.read_json('../data/lewtun-drug-reviews/test.jsonl', lines=True)

# Extract a random sample of 1000 from the data sets
train_sample = train_df.sample(n=1000)
test_sample = test_df.sample(n=1000)

# Write the samples to csv
train_sample.to_json('../data/lewtun_train_sample.jsonl', lines=True, orient='records')
test_sample.to_json('../data/lewtun_test_sample.jsonl', lines=True, orient='records')