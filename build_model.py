from typing import Dict, Tuple

import tensorflow as tf
from tensorflow.keras import layers


def build_model(embedding_matrix, max_lens: Dict, tokenizer_len: int = 100):
    input_text = layers.Input(shape=(max_lens['text'], ), name='text')
    embedding_text = layers.Embedding(tokenizer_len,
                                      100,
                                      weights=[embedding_matrix],
                                      input_length=max_lens['text'],
                                      trainable=False)(input_text)
    dropout_text = layers.Dropout(0.2)(embedding_text)
    lstm_text_1 = layers.LSTM(128, return_sequences=True)(dropout_text)
    lstm_text_2 = layers.LSTM(128, return_sequences=True)(lstm_text_1)
    lstm_text_3 = layers.LSTM(128, return_sequences=True)(lstm_text_2)
    lstm_text_4 = layers.LSTM(128)(lstm_text_3)

    input_ky = layers.Input(shape=(max_lens['keyword'], ), name='keyword')
    embedding_ky = layers.Embedding(tokenizer_len,
                                    100,
                                    weights=[embedding_matrix],
                                    input_length=max_lens['keyword'],
                                    trainable=False)(input_ky)
    dropout_ky = layers.Dropout(0.2)(embedding_ky)
    lstm_ky_1 = layers.LSTM(64, return_sequences=True)(dropout_ky)
    lstm_ky_2 = layers.LSTM(64, return_sequences=True)(lstm_ky_1)
    lstm_ky_3 = layers.LSTM(64, return_sequences=True)(lstm_ky_2)
    lstm_ky_4 = layers.LSTM(64)(lstm_ky_3)

    input_loc = layers.Input(shape=(max_lens['location'], ), name='location')
    embedding_loc = layers.Embedding(tokenizer_len,
                                     100,
                                     weights=[embedding_matrix],
                                     input_length=max_lens['location'],
                                     trainable=False)(input_loc)
    dropout_loc = layers.Dropout(0.2)(embedding_loc)
    lstm_loc_1 = layers.LSTM(32, return_sequences=True)(dropout_loc)
    lstm_loc_2 = layers.LSTM(32, return_sequences=True)(lstm_loc_1)
    lstm_loc_3 = layers.LSTM(32, return_sequences=True)(lstm_loc_2)
    lstm_loc_4 = layers.LSTM(32)(lstm_loc_3)

    merge = layers.concatenate([lstm_text_4, lstm_ky_4, lstm_loc_4])

    dropout = layers.Dropout(0.5)(merge)
    dense1 = layers.Dense(256, activation='relu')(dropout)
    dense2 = layers.Dense(128, activation='relu')(dense1)
    output = layers.Dense(2, activation='softmax')(dense2)

    model = tf.keras.Model(inputs={
        'text': input_text,
        'keyword': input_ky,
        'location': input_loc
    },
                           outputs=output)
    tf.keras.utils.plot_model(model,
                              to_file="./tmp/model.png",
                              show_shapes=True)
    return model