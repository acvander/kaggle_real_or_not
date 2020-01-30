import re

import pandas as pd


def extract_hashtags(df: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(r'#(\w+)')

    def get_hashtags(row: pd.Series) -> pd.Series:
        text = row['text']
        hashtags = re.findall(pattern, text)
        row['hashtags'] = hashtags

        return row

    df = df.apply(get_hashtags, axis=1)
    return df