from typing import Dict, Tuple
import pandas as pd

from preprocessing.extract_hashtags import extract_hashtags


def clean_row(row: pd.Series):
    # remove %20
    row['keyword'] = row['keyword'].replace('%20', ' ')
    return row


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # replace NaNs with empty string
    df = df.fillna(value='')
    clean_df = df.apply(clean_row, axis=1)
    return clean_df


def preprocess(train_df: pd.DataFrame,
               test_df: pd.DataFrame,
               num_words: int = 10000):
    clean_train_df = clean_data(train_df)
    clean_test_df = clean_data(test_df)

    clean_train_df = extract_hashtags(clean_train_df)
    clean_test_df = extract_hashtags(clean_test_df)

    clean_train_df.to_csv('./tmp/train.csv', index=False)
    clean_test_df.to_csv('./tmp/test.csv', index=False)
