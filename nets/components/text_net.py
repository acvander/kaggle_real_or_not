from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def _add_multiple_lstms(input_layer, num_lstms: int, size: int):
    lstm_layers = []
    # initial layer
    lstm_layers.append(layers.LSTM(size, return_sequences=True)(input_layer))
    for i in range(1, num_lstms - 1):
        lstm_layers.append(
            layers.LSTM(size, return_sequences=True)(lstm_layers[-1]))
    # last layer
    output_layer = layers.LSTM(size)(lstm_layers[-1])
    return output_layer


def text_net(embedding_matrix,
             max_lens: Dict,
             tokenizer_len: int = 100,
             net_scale: int = 64):
    embedding_size = 100
    embedding_dropout = 0.5
    mask_zero = False
    lstm_depth = 2

    base_sizes = np.array([4, 2, 1])
    lstm_sizes = base_sizes * net_scale
    input_text = layers.Input(shape=(max_lens['text'], ), name='text')
    embedding_text = layers.Embedding(tokenizer_len,
                                      embedding_size,
                                      weights=[embedding_matrix],
                                      input_length=max_lens['text'],
                                      trainable=False,
                                      mask_zero=mask_zero)(input_text)
    dropout_text = layers.Dropout(embedding_dropout)(embedding_text)
    lstm_text = _add_multiple_lstms(dropout_text, lstm_depth, lstm_sizes[0])

    input_ky = layers.Input(shape=(max_lens['keyword'], ), name='keyword')
    embedding_ky = layers.Embedding(tokenizer_len,
                                    embedding_size,
                                    weights=[embedding_matrix],
                                    input_length=max_lens['keyword'],
                                    trainable=False,
                                    mask_zero=mask_zero)(input_ky)
    dropout_ky = layers.Dropout(embedding_dropout)(embedding_ky)
    lstm_ky = _add_multiple_lstms(dropout_ky, lstm_depth, lstm_sizes[1])

    input_loc = layers.Input(shape=(max_lens['location'], ), name='location')
    embedding_loc = layers.Embedding(tokenizer_len,
                                     embedding_size,
                                     weights=[embedding_matrix],
                                     input_length=max_lens['location'],
                                     trainable=False,
                                     mask_zero=mask_zero)(input_loc)
    dropout_loc = layers.Dropout(embedding_dropout)(embedding_loc)
    lstm_loc = _add_multiple_lstms(dropout_loc, lstm_depth, lstm_sizes[2])

    # hashtag branch
    input_hashtag = layers.Input(shape=(max_lens['hashtags'], ),
                                 name='hashtags')
    embedding_hashtag = layers.Embedding(tokenizer_len,
                                         embedding_size,
                                         weights=[embedding_matrix],
                                         input_length=max_lens['hashtags'],
                                         trainable=False,
                                         mask_zero=mask_zero)(input_hashtag)
    dropout_hashtag = layers.Dropout(embedding_dropout)(embedding_hashtag)
    lstm_hashtag = _add_multiple_lstms(dropout_hashtag, lstm_depth,
                                       lstm_sizes[1])

    merge = layers.concatenate([lstm_text, lstm_ky, lstm_loc, lstm_hashtag])
    input_layers = [input_text, input_loc, input_ky, input_hashtag]
    return (input_layers, merge)