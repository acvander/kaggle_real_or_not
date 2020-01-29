from typing import Dict

import numpy as np
import tensorflow as tf

from tensorflow_addons.metrics import FBetaScore, F1Score

from build_model import build_model


def train_model(train_data: Dict,
                embeddings: np.array,
                pad_lens: Dict,
                num_words: int = 1000,
                model_path: str = './tmp/model.h5') -> tf.keras.models.Model:
    model = build_model(
        embeddings,
        pad_lens,
        tokenizer_len=num_words,
    )

    metrics = []
    # f1score = CustomF1Score()
    # metrics.append(f1score)
    metrics.append(tf.keras.metrics.Recall())
    # metrics.append(FBetaScore(1))
    metrics.append(F1Score(2, average='micro'))
    model.compile(loss="binary_crossentropy",
                  optimizer='adam',
                  metrics=metrics)

    callbacks = []
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path,
                                                    'val_f1_score',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode='max')
    callbacks.append(checkpoint)

    model.fit(train_data,
              train_data['target'],
              validation_split=0.2,
              epochs=25,
              batch_size=128,
              verbose=1,
              shuffle=True,
              callbacks=callbacks)

    model.save('./tmp/model.h5')

    return model