from datasets import load_dataset

# Load the dataset
dataset_laroseda = load_dataset("universityofbucharest/laroseda")


import pandas as pd

train_df = dataset_laroseda["train"].to_pandas()
test_df = dataset_laroseda["test"].to_pandas()

# Save them separately
train_df.to_csv("laroseda_train.csv", index=False)
test_df.to_csv("laroseda_test.csv", index=False)

# Or combine into one CSV if you want everything in a single file
all_df = pd.concat([train_df, test_df], ignore_index=True)
all_df.to_csv("laroseda_all.csv", index=False)
