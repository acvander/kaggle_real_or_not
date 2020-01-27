from typing import Sequence
import numpy as np
import pandas as pd
import tensorflow as tf

TRAIN_PATH = './data/train.csv'
TEST_PATH = './data/test.csv'


def clean_row(row: pd.Series):
    # remove %20
    row['keyword'] = row['keyword'].replace('%20', ' ')
    return row


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # replace NaNs with empty string
    df = df.fillna(value='')
    clean_df = df.apply(clean_row, axis=1)
    return clean_df


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    columns = train_df.columns.to_list()
    example = train_df.iloc[100].to_list()

    clean_train_df = clean_data(train_df)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=35000,
                                                      lower=True)
    tokenizer.fit_on_texts(
        (clean_train_df['keyword'] + clean_train_df['location'] +
         clean_train_df['text'].to_list()))
    return


if __name__ == "__main__":
    main()