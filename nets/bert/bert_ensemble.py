import os
import shelve
from pathlib import Path

import numpy as np

from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.utils import plot_model

from tensorflow_hub import KerasLayer
from absl import logging
from sklearn.metrics import f1_score


def create_bert_ensemble(model_dir: str = './tmp/debug_bert',
                         model_name: str = 'bert'):
    '''won't work on my GPU due to memory limitations'''
    sub_models = []

    # get names of all models in directory
    model_files = Path(model_dir).glob('*.h5')

    for model in model_files:

        # remove initial_weights
        if model.stem == 'initial_weights':
            continue
        # remove any existing ensemble
        if model.stem == 'ensemble':
            continue
        # load models
        logging.info('loading {}'.format(model.stem))
        sub_model = load_model(model,
                               compile=False,
                               custom_objects={'KerasLayer': KerasLayer})
        sub_models.append(sub_model)

    text_input = layers.Input(shape=(sub_models[0].input))


def eval_bert_ensemble(model_dir: str = './tmp/debug_bert',
                       model_name: str = 'bert',
                       batch_size: int = 32):

    # load data
    logging.info('loading data')
    with shelve.open('./tmp/bert_data/bert_shelf') as shelf:
        train_input = shelf['train_input']
        train_output = shelf['train_output']
        test_input = shelf['test_input']

    sub_models = []

    # get names of all models in directory
    model_files = Path(model_dir).glob('*.h5')
    weight_files = Path(model_dir).glob('*_weights.h5')
    train_predictions = []
    test_predictions = []

    model_path = list(Path(model_dir).glob('*_0.h5'))[0]
    model = load_model(model_path,
                       compile=False,
                       custom_objects={'KerasLayer': KerasLayer})

    for weights in weight_files:
        # remove initial_weights
        if weights.stem == 'initial_weights':
            continue

        # load models
        logging.info('loading {}'.format(weights.stem))
        model.load_weights(str(weights))
        logging.info('predicting on train data')
        train_pred = model.predict(train_input,
                                   verbose=1,
                                   batch_size=batch_size)
        train_predictions.append(train_pred)

        logging.info('predicting on test data')

        test_pred = model.predict(test_input, verbose=1, batch_size=batch_size)
        test_predictions.append(test_pred)

    # average predictions
    avg = np.mean(train_predictions, axis=0)
    pred = np.argmax(avg, axis=1)
    train_f1 = f1_score(np.argmax(train_output, axis=1), pred)
    print('train f1_score: {}'.format(train_f1))

    avg = np.mean(test_predictions, axis=0)
    pred = np.argmax(avg, axis=1)
    return
