from typing import Dict, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers


def load_embeddings(tokenizer: Tokenizer,
                    embeddings_file: str = './data/glove.6B.100d.txt',
                    embedding_dim: int = 100,
                    max_words: int = 1000) -> np.array:
    embedding_dict = {}
    with open(embeddings_file) as glove_file:
        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = np.asarray(records[1:], dtype='float32')
            embedding_dict[word] = vector_dimensions

    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, idx in tokenizer.word_index.items():
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector

    return embedding_matrix


def prepare_data(train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 num_words=1000) -> Tuple[Dict, Dict, Tokenizer, Dict]:
    # hashtags = train_df['hashtags'].to_list()
    # fill in nan
    train_df = train_df.fillna(value='')
    test_df = test_df.fillna(value='')

    text_lengths = [len(x.split()) for x in train_df['text'].to_list()]
    kw_lengths = [len(x.split()) for x in train_df['keyword'].to_list()]
    loc_lengths = [len(x.split()) for x in train_df['location'].to_list()]
    hashtag_lengths = [len(x) for x in train_df['hashtags'].to_list()]

    max_text_len = max(text_lengths) + 1
    max_kw_len = max(kw_lengths) + 1
    max_loc_len = max(loc_lengths) + 1
    max_hashtag_len = max(hashtag_lengths) + 1

    # tokenize data
    tokenizer = Tokenizer(num_words=num_words, lower=True)
    tokenizer.fit_on_texts((train_df['keyword'] + train_df['location'] +
                            train_df['text'].to_list()))

    train_text = tokenizer.texts_to_sequences(train_df['text'])
    train_keyword = tokenizer.texts_to_sequences(train_df['keyword'])
    train_location = tokenizer.texts_to_sequences(train_df['location'])
    train_hashtag = tokenizer.texts_to_sequences(train_df['hashtags'])
    train_id = train_df['id'].to_numpy()

    test_text = tokenizer.texts_to_sequences(test_df['text'])
    test_keyword = tokenizer.texts_to_sequences(test_df['keyword'])
    test_location = tokenizer.texts_to_sequences(test_df['location'])
    test_hashtag = tokenizer.texts_to_sequences(test_df['hashtags'])
    test_id = test_df['id'].to_numpy()

    label = train_df['target'].to_list()
    label = tf.keras.utils.to_categorical(label)

    # pad sequences
    train_text = pad_sequences(train_text, maxlen=max_text_len)
    train_keyword = pad_sequences(train_keyword, maxlen=max_kw_len)
    train_location = pad_sequences(train_location, maxlen=max_loc_len)
    train_hashtags = pad_sequences(train_hashtag, maxlen=max_hashtag_len)

    test_text = pad_sequences(test_text, maxlen=max_text_len)
    test_keyword = pad_sequences(test_keyword, maxlen=max_kw_len)
    test_location = pad_sequences(test_location, maxlen=max_loc_len)
    test_hashtags = pad_sequences(test_hashtag, maxlen=max_hashtag_len)

    train_data = {
        'text': train_text,
        'keyword': train_keyword,
        'location': train_location,
        'hashtags': train_hashtags,
        'target': label,
        'id': train_id
    }
    test_data = {
        'text': test_text,
        'keyword': test_keyword,
        'location': test_location,
        'hashtags': test_hashtags,
        'id': test_id
    }
    pad_lens = {
        'text': max_text_len,
        'keyword': max_kw_len,
        'location': max_loc_len,
        'hashtags': max_hashtag_len
    }

    embeddings = load_embeddings(tokenizer, max_words=num_words)

    return (train_data, test_data, embeddings, pad_lens)
