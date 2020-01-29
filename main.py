from preprocessing import preprocess
from metrics.CustomF1 import CustomF1Score
from typing import Dict, Sequence, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers

from tensorflow_addons.metrics import FBetaScore, F1Score

from build_model import build_model

TRAIN_PATH = './data/train.csv'
TEST_PATH = './data/test.csv'
NUM_WORDS = 35000
MODEL_PATH = './tmp/model.h5'


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    columns = train_df.columns.to_list()
    example = train_df.iloc[100].to_list()

    train_data, test_data, embeddings, pad_lens = preprocess(
        train_df, test_df, num_words=NUM_WORDS)

    model = build_model(
        embeddings,
        pad_lens,
        tokenizer_len=NUM_WORDS,
    )

    metrics = []
    # f1score = CustomF1Score()
    # metrics.append(f1score)
    metrics.append(tf.keras.metrics.Recall())
    # metrics.append(FBetaScore(1))
    # metrics.append(F1Score(2))
    model.compile(loss="binary_crossentropy",
                  optimizer='adam',
                  metrics=metrics)

    callbacks = []
    checkpoint = tf.keras.callbacks.ModelCheckpoint(MODEL_PATH,
                                                    'val_recall',
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
              callbacks=callbacks)

    model.save('./tmp/model.h5')
    return


if __name__ == "__main__":
    main()