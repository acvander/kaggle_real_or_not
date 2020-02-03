import tensorflow as tf

from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


def build_bert_model(bert_layer, max_len=512):
    def inner_build_model():
        input_word_ids = Input(shape=(max_len, ),
                               dtype=tf.int32,
                               name='input_word_ids')
        input_mask = Input(shape=(max_len, ),
                           dtype=tf.int32,
                           name='input_mask')
        segment_ids = Input(shape=(max_len, ),
                            dtype=tf.int32,
                            name='segment_ids')

        _, sequence_output = bert_layer(
            [input_word_ids, input_mask, segment_ids])
        clf_output = Bidirectional(LSTM(128))(sequence_output)
        #         clf_output = sequence_output[:, 0, :]
        out = Dense(1, activation='sigmoid')(clf_output)

        model = Model(inputs={
            'word_ids': input_word_ids,
            'mask': input_mask,
            'segment_ids': segment_ids
        },
                      outputs=out)
        model.compile(Adam(lr=2e-6),
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'mse'])

        return model

    return inner_build_model