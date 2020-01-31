from nets.resnet_model import resnet_model
from nets import base_model
from typing import Dict

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow_addons.metrics import F1Score


def _plot_training_data(history: tf.keras.callbacks.History):
    fig, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].plot(history['f1_score'], 'b')
    axs[0].plot(history['val_f1_score'], 'r')
    axs[0].set_title('F1 Score')
    axs[0].set_ylim(bottom=0.5, top=1)

    axs[1].plot(history['loss'], 'b')
    axs[1].plot(history['val_loss'], 'r')
    axs[1].set_title('Loss')
    axs[1].set_ylim(bottom=0, top=1)

    plt.tight_layout()
    plt.savefig('./tmp/training_history.png')
    return


def train_model(train_data: Dict,
                embeddings: np.array,
                pad_lens: Dict,
                num_words: int = 1000,
                model_path: str = './tmp/model.h5',
                epochs: int = 25,
                net_scale: int = 64,
                learn_rate: float = 0.001) -> tf.keras.models.Model:
    model = resnet_model(embeddings,
                         pad_lens,
                         tokenizer_len=num_words,
                         net_scale=net_scale)

    metrics = []
    # f1score = CustomF1Score()
    # metrics.append(f1score)
    # metrics.append(tf.keras.metrics.Recall())
    # metrics.append(FBetaScore(1))
    metrics.append(F1Score(2, average='micro'))

    optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=metrics)

    callbacks = []
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path,
                                                    'val_f1_score',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode='max')
    callbacks.append(checkpoint)

    early_stop = tf.keras.callbacks.EarlyStopping('val_f1_score',
                                                  min_delta=0.01,
                                                  mode='max',
                                                  patience=25,
                                                  verbose=1)
    callbacks.append(early_stop)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau('val_f1_score',
                                                     factor=0.1,
                                                     min_delta=0.01,
                                                     mode='max',
                                                     patience=8,
                                                     min_lr=learn_rate *
                                                     10**-3,
                                                     verbose=1)
    callbacks.append(reduce_lr)

    history = model.fit(train_data,
                        train_data['target'],
                        validation_split=0.2,
                        epochs=epochs,
                        batch_size=128,
                        verbose=1,
                        shuffle=True,
                        callbacks=callbacks)

    model.save('./tmp/model.h5')
    _plot_training_data(history.history)

    return model