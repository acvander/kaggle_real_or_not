import os
from typing import Dict, List
from utils.plot.plot import plot_k_fold_data, plot_training_data

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow_addons.metrics import F1Score

from preprocessing.k_folds import k_fold_split
from nets.resnet_model import resnet_model
from nets.avg_ensemble import avg_ensemble
from nets import base_model


def compile_model(model,
                  learn_rate: float = 0.001,
                  model_path: str = './tmp/model.h5'):
    metrics = []
    metrics.append(F1Score(2, average='micro'))

    optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=metrics)

    return model


def create_callbacks(learn_rate: float = 0.001,
                     model_path: str = './tmp/model.h5',
                     epochs: int = 25):
    callbacks = []
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path,
                                                    'val_f1_score',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode='max')
    callbacks.append(checkpoint)

    early_stop_patience = epochs // 3
    early_stop = tf.keras.callbacks.EarlyStopping('val_f1_score',
                                                  min_delta=0.01,
                                                  mode='max',
                                                  patience=early_stop_patience,
                                                  verbose=1)
    callbacks.append(early_stop)

    reduce_lr_patience = early_stop_patience // 2 - 1
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        'val_f1_score',
        factor=0.1,
        min_delta=0.01,
        mode='max',
        patience=reduce_lr_patience,
        min_lr=learn_rate * 10**-3,
        verbose=1)
    callbacks.append(reduce_lr)
    return callbacks


def train_k_folds(train_data: Dict,
                  embeddings: np.array,
                  pad_lens: Dict,
                  num_words: int = 1000,
                  model_dir: str = './tmp/model/',
                  model_name: str = 'model',
                  fig_name: str = 'history',
                  epochs: int = 25,
                  net_scale: int = 64,
                  learn_rate: float = 0.001) -> tf.keras.models.Model:

    # create dir if needed
    os.makedirs(model_dir, exist_ok=True)

    k_folds = k_fold_split(train_data, shuffle=True)
    histories = []
    for i, (valid_idxs, train_idxs) in enumerate(k_folds):
        print('training fold #{}'.format(i + 1))
        model_path = '{}_{}.h5'.format(os.path.join(model_dir, model_name), i)
        train_fold_data = {
            key: val[train_idxs]
            for (key, val) in train_data.items()
        }
        valid_fold_data = {
            key: val[valid_idxs]
            for (key, val) in train_data.items()
        }

        model = resnet_model(embeddings,
                             pad_lens,
                             tokenizer_len=num_words,
                             net_scale=net_scale,
                             fig_path=os.path.join(model_dir,
                                                   'model_{}.png'.format(i)))

        model = compile_model(model, learn_rate=learn_rate)
        callbacks = create_callbacks(model_path=model_path, epochs=epochs)

        history = model.fit(train_fold_data,
                            train_fold_data['target'],
                            validation_data=(valid_fold_data,
                                             valid_fold_data['target']),
                            epochs=epochs,
                            batch_size=128,
                            verbose=1,
                            shuffle=True,
                            callbacks=callbacks)

        model.save(model_path)
        histories.append(history)
        plot_training_data(
            history.history,
            '{}_{}.png'.format(os.path.join(model_dir, fig_name), i))
    plot_k_fold_data(
        histories, '{}_combined.png'.format(os.path.join(model_dir, fig_name)))

    ensemble_model = avg_ensemble(model_dir)
    return ensemble_model