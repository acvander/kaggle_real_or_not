import os
import shelve
from utils.plot.plot import plot_k_fold_data, plot_training_data

from absl import logging
import tensorflow as tf
from tensorflow.keras.models import clone_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import tensorflow_hub as hub
from sklearn.model_selection import StratifiedKFold
from tensorflow_addons.metrics import F1Score

from nets.bert_model import build_bert_model


def compile_model(model,
                  learn_rate: float = 0.001,
                  model_path: str = './tmp/model.h5'):
    metrics = []
    metrics.append(F1Score(2, average='micro'))

    optimizer = Adam(lr=learn_rate)
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=metrics)

    return


def create_callbacks(learn_rate: float = 0.001,
                     model_path: str = './tmp/model.h5',
                     epochs: int = 25):
    callbacks = []
    checkpoint = ModelCheckpoint(model_path,
                                 'val_f1_score',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    callbacks.append(checkpoint)

    # early_stop_patience = epochs // 3
    # early_stop = EarlyStopping('val_f1_score',
    #                            min_delta=0.01,
    #                            mode='max',
    #                            patience=early_stop_patience,
    #                            verbose=1)
    # callbacks.append(early_stop)

    # reduce_lr_patience = early_stop_patience // 2 - 1
    # reduce_lr = ReduceLROnPlateau('val_f1_score',
    #                               factor=0.1,
    #                               min_delta=0.01,
    #                               mode='max',
    #                               patience=reduce_lr_patience,
    #                               min_lr=learn_rate * 10**-3,
    #                               verbose=1)
    # callbacks.append(reduce_lr)
    return callbacks


def train_bert(model_dir: str = './tmp/bert_default',
               model_name: str = 'bert_default',
               shuffle: bool = True,
               fig_name: str = 'bert',
               epochs: int = 3):
    # load data
    logging.info('loading data')
    with shelve.open('./tmp/bert_data/bert_shelf') as shelf:
        train_input = shelf['train_input']
        train_output = shelf['train_output']
        test_input = shelf['test_input']
        max_token_len = shelf['max_token_len']
        bert_url = shelf['bert_url']

    logging.info('loading model')

    os.makedirs(model_dir, exist_ok=True)

    splitter = StratifiedKFold(n_splits=3, shuffle=shuffle, random_state=2020)
    x = range(len(train_output))
    k_folds = list(splitter.split(
        x, train_output[:, 0]))  # only need to use one value for grouping

    histories = []
    for i, (valid_idxs, train_idxs) in enumerate(k_folds):
        logging.info('training fold #{}'.format(i + 1))

        bert_layer = hub.KerasLayer(bert_url, trainable=True)
        model = build_bert_model(bert_layer, max_len=max_token_len)()

        model_path = os.path.join(model_dir, '{}_{}.h5'.format(model_name, i))

        # compile modelf
        compile_model(model, learn_rate=2e-6, model_path=model_path)

        # separate data
        fold_train_input = {
            key: val[train_idxs]
            for (key, val) in train_input.items()
        }
        fold_train_output = train_output[train_idxs]
        fold_valid_input = {
            key: val[valid_idxs]
            for (key, val) in train_input.items()
        }
        fold_valid_output = train_output[valid_idxs]

        callbacks = create_callbacks(model_path=model_path)

        history = model.fit(fold_train_input,
                            fold_train_output,
                            validation_data=(fold_valid_input,
                                             fold_valid_output),
                            epochs=epochs,
                            batch_size=8,
                            callbacks=callbacks)
        plot_training_data(
            history.history,
            '{}_{}.png'.format(os.path.join(model_dir, fig_name), i))
        histories.append(history)

        del model
        tf.keras.backend.clear_session()
    plot_k_fold_data(
        histories, '{}_combined.png'.format(os.path.join(model_dir, fig_name)))

    return
