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

from build_model import build_model

TRAIN_PATH = './data/train.csv'
TEST_PATH = './data/test.csv'
NUM_WORDS = 35000
MODEL_PATH = './tmp/model.h5'


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_data, test_data, embeddings, pad_lens = preprocess(
        train_df, test_df, num_words=NUM_WORDS)

    model = train_model(train_data,
                        embeddings,
                        pad_lens,
                        num_words=NUM_WORDS,
                        model_path=MODEL_PATH)
    return


if __name__ == "__main__":
    main()