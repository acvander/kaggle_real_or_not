import os
import shelve

from absl import logging
from tensorflow.keras.utils import plot_model
import tensorflow_hub as hub

from nets.bert_model import build_bert_model


def train_bert(model_dir: str = './tmp/bert_default',
               model_name: str = 'bert_default'):
    # load data
    logging.info('loading data')
    with shelve.open('./tmp/bert_data/bert_shelf') as shelf:
        train_input = shelf['train_input']
        train_output = shelf['train_output']
        test_input = shelf['test_input']
        max_token_len = shelf['max_token_len']
        bert_url = shelf['bert_url']

    logging.info('loading model')
    bert_layer = hub.KerasLayer(bert_url, trainable=True)

    os.makedirs(model_dir, exist_ok=True)

    model = build_bert_model(bert_layer, max_len=max_token_len)()
    plot_model(model,
               to_file=os.path.join(model_dir, model_name + '.png'),
               show_shapes=True)

    model.fit(train_input, train_output, epochs=5, batch_size=4)

    return
