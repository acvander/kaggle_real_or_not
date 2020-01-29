from train_model import train_model
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

from absl import app, flags

from build_model import build_model

FLAGS = flags.FLAGS

flags.DEFINE_string('train_path', './data/train.csv', 'path of train.csv')
flags.DEFINE_string('test_path', './data/test.csv', 'path of train.csv')
flags.DEFINE_integer('num_words', 35000, 'number of words to use')
flags.DEFINE_string('model_path', './tmp/model.h5', 'path to save model')


def main(argv):
    train_df = pd.read_csv(FLAGS.train_path)
    test_df = pd.read_csv(FLAGS.test_path)

    train_data, test_data, embeddings, pad_lens = preprocess(
        train_df, test_df, num_words=FLAGS.num_words)

    model = train_model(train_data,
                        embeddings,
                        pad_lens,
                        num_words=FLAGS.num_words,
                        model_path=FLAGS.model_path)
    return


if __name__ == "__main__":
    app.run(main)