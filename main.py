from preprocessing import prepare_data
from typing import Dict, Sequence, Tuple
from ast import literal_eval

import pandas as pd

from absl import app, flags

from train_model import train_model
from preprocessing import preprocess, prepare_data
from build_model import build_model
from gen_submission import gen_submission

FLAGS = flags.FLAGS

flags.DEFINE_enum('mode', 'gen_submission',
                  ['preprocess', 'train', 'gen_submission'],
                  'defines mode to run app')
flags.DEFINE_string('train_path', './data/train.csv', 'path of train.csv')
flags.DEFINE_string('test_path', './data/test.csv', 'path of train.csv')
flags.DEFINE_integer('num_words', 35000, 'number of words to use')
flags.DEFINE_string('model_path', './tmp/model.h5', 'path to save model')


def main(argv):

    if FLAGS.mode == 'preprocess':
        train_df = pd.read_csv(FLAGS.train_path)
        test_df = pd.read_csv(FLAGS.test_path)
        preprocess(train_df, test_df, num_words=FLAGS.num_words)
    elif FLAGS.mode == 'train':
        train_df = pd.read_csv('./tmp/train.csv',
                               converters={"hashtags": literal_eval})
        test_df = pd.read_csv('./tmp/test.csv')
        train_data, test_data, embeddings, pad_lens = prepare_data(
            train_df, test_df, num_words=FLAGS.num_words)
        model = train_model(train_data,
                            embeddings,
                            pad_lens,
                            num_words=FLAGS.num_words,
                            model_path=FLAGS.model_path)
    elif FLAGS.mode == 'gen_submission':
        train_df = pd.read_csv('./tmp/train.csv',
                               converters={"hashtags": literal_eval})
        test_df = pd.read_csv('./tmp/test.csv')
        train_data, test_data, embeddings, pad_lens = prepare_data(
            train_df, test_df, num_words=FLAGS.num_words)
        gen_submission(test_data, model_path=FLAGS.model_path)
    else:
        raise Exception('improper selection for mode')

    return


if __name__ == "__main__":
    app.run(main)