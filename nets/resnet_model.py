from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def resnet_model(embedding_matrix,
                 max_lens: Dict,
                 tokenizer_len: int = 100,
                 net_scale: int = 64) -> tf.keras.Model:
    embedding_size = 100
    embedding_dropout = 0.5
    mask_zero = False

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
    lstm_text_1 = layers.LSTM(lstm_sizes[0],
                              return_sequences=True)(dropout_text)
    lstm_text_2 = layers.LSTM(lstm_sizes[0],
                              return_sequences=True)(lstm_text_1)
    lstm_text_3 = layers.LSTM(lstm_sizes[0],
                              return_sequences=True)(lstm_text_2)
    lstm_text_4 = layers.LSTM(lstm_sizes[0])(lstm_text_3)

    input_ky = layers.Input(shape=(max_lens['keyword'], ), name='keyword')
    embedding_ky = layers.Embedding(tokenizer_len,
                                    embedding_size,
                                    weights=[embedding_matrix],
                                    input_length=max_lens['keyword'],
                                    trainable=False,
                                    mask_zero=mask_zero)(input_ky)
    dropout_ky = layers.Dropout(embedding_dropout)(embedding_ky)
    lstm_ky_1 = layers.LSTM(lstm_sizes[1], return_sequences=True)(dropout_ky)
    lstm_ky_2 = layers.LSTM(lstm_sizes[1], return_sequences=True)(lstm_ky_1)
    lstm_ky_3 = layers.LSTM(lstm_sizes[1], return_sequences=True)(lstm_ky_2)
    lstm_ky_4 = layers.LSTM(lstm_sizes[1])(lstm_ky_3)

    input_loc = layers.Input(shape=(max_lens['location'], ), name='location')
    embedding_loc = layers.Embedding(tokenizer_len,
                                     embedding_size,
                                     weights=[embedding_matrix],
                                     input_length=max_lens['location'],
                                     trainable=False,
                                     mask_zero=mask_zero)(input_loc)
    dropout_loc = layers.Dropout(embedding_dropout)(embedding_loc)
    lstm_loc_1 = layers.LSTM(lstm_sizes[2], return_sequences=True)(dropout_loc)
    lstm_loc_2 = layers.LSTM(lstm_sizes[2], return_sequences=True)(lstm_loc_1)
    lstm_loc_3 = layers.LSTM(lstm_sizes[2], return_sequences=True)(lstm_loc_2)
    lstm_loc_4 = layers.LSTM(lstm_sizes[2])(lstm_loc_3)

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
    lstm_hashtag_1 = layers.LSTM(lstm_sizes[1],
                                 return_sequences=True)(dropout_hashtag)
    lstm_hashtag_2 = layers.LSTM(lstm_sizes[1],
                                 return_sequences=True)(lstm_hashtag_1)
    lstm_hashtag_3 = layers.LSTM(lstm_sizes[1],
                                 return_sequences=True)(lstm_hashtag_2)
    lstm_hashtag_4 = layers.LSTM(lstm_sizes[1])(lstm_hashtag_3)

    merge = layers.concatenate(
        [lstm_text_4, lstm_ky_4, lstm_loc_4, lstm_hashtag_4])

    dropout = layers.Dropout(0.5)(merge)
    dense1 = layers.Dense(1024, activation='relu')(dropout)
    res1 = layers.concatenate([merge, dense1])

    dropout2 = layers.Dropout(0.5)(res1)
    dense2 = layers.Dense(512, activation='relu')(dropout2)
    res2 = layers.concatenate([res1, dense2])

    output = layers.Dense(2, activation='softmax')(res2)

    model = tf.keras.Model(inputs={
        'text': input_text,
        'keyword': input_ky,
        'location': input_loc,
        'hashtags': input_hashtag
    },
                           outputs=output)
    tf.keras.utils.plot_model(model,
                              to_file="./tmp/model.png",
                              show_shapes=True)
    return model