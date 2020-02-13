import tensorflow as tf

from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Activation
from tensorflow.keras import Model


def build_bert_model(bert_layer, max_len=512) -> Model:
    '''adapted from https://www.kaggle.com/lhideki/bert-with-kfold'''

    input_word_ids = Input(shape=(max_len, ), dtype=tf.int32, name='tokens')
    input_mask = Input(shape=(max_len, ), dtype=tf.int32, name='masks')
    segment_ids = Input(shape=(max_len, ), dtype=tf.int32, name='segments')

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = Bidirectional(LSTM(128))(sequence_output)
    #         clf_output = sequence_output[:, 0, :]
    dense = Dense(2)(clf_output)
    out = Activation('softmax')(dense)

    model = Model(inputs={
        'word_ids': input_word_ids,
        'mask': input_mask,
        'segment_ids': segment_ids
    },
                  outputs=out)

    return model
