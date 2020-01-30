from typing import Dict, Tuple
import re
import pandas as pd

from preprocessing.extract_hashtags import extract_hashtags


def clean_text(text):
    # this symbol should be an apostrophe
    text = re.sub(r'\x89Ûª', '\'', text)
    # not sure what this one is for
    text = re.sub(r'\x89Û\w', '', text)
    text = re.sub(r'Û÷', '', text)
    text = re.sub(r'Û¢', '', text)
    text = re.sub(r'å©', '', text)

    # remove any links
    text = re.sub(r'https?://\S+', '', text)
    # remove apostrophes used in a contraction and combine letters
    # text=re.sub(r'\')

    # remove punctuation
    text = re.sub(r'[#.,!$%^&*?;@]', ' ', text)
    # escaped punctuation
    text = re.sub(r'[\(\)\[\]\'\"\-\/\|\:\_]', ' ', text)
    # remove new lines and extra spaces
    # try to keep this one last to remove any weird punctuation leftovers
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s\s+', ' ', text)
    return text


def clean_row(row: pd.Series):
    text = row['text']
    text = clean_text(text)

    row['text'] = text

    # cleanup locations
    row['location'] = clean_text(row['location'])

    # remove %20 from keywords
    row['keyword'] = row['keyword'].replace('%20', ' ')
    row['keyword'] = clean_text(row['keyword'])
    return row


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # replace NaNs with empty string
    df = df.fillna(value='')
    clean_df = df.apply(clean_row, axis=1)
    return clean_df


def preprocess(train_df: pd.DataFrame,
               test_df: pd.DataFrame,
               num_words: int = 10000):

    clean_train_df = extract_hashtags(train_df)
    clean_test_df = extract_hashtags(test_df)

    clean_train_df = clean_data(clean_train_df)
    clean_test_df = clean_data(clean_test_df)

    clean_train_df.to_csv('./tmp/train.csv', index=False)
    clean_test_df.to_csv('./tmp/test.csv', index=False)
