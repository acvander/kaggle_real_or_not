from training.train_k_folds import train_k_folds
from typing import Dict, Sequence, Tuple
from ast import literal_eval

import pandas as pd
import tensorflow as tf

from absl import app, flags

from training import train_model, train_k_folds
from preprocessing import preprocess, prepare_data
from gen_submission import gen_submission
from preprocessing.k_folds import k_fold_split

# set logging level
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

FLAGS = flags.FLAGS

flags.DEFINE_enum('mode', 'gen_submission',
                  ['preprocess', 'train', 'train_k_fold', 'gen_submission'],
                  'defines mode to run app')
flags.DEFINE_string('train_path', './data/train.csv', 'path of train.csv')
flags.DEFINE_string('test_path', './data/test.csv', 'path of train.csv')
flags.DEFINE_integer('num_words', 35000, 'number of words to use')
flags.DEFINE_string('model_dir', './tmp/model/',
                    'directory to which to save model')
flags.DEFINE_string('model_name', 'model', 'model name')
flags.DEFINE_integer('epochs', 25, 'number of training epochs')
flags.DEFINE_integer('net_scale', 64, 'scaling for network sizes')
flags.DEFINE_float('subset', 1.0, 'subset ratio of data')


def subset_data(data: Dict, subset_ratio: float = 1.0):
    new_data = {}
    for (key, val) in data.items():
        data_len = len(val)
        subset_len = int(data_len * subset_ratio)
        new_data[key] = val[:subset_len]
    return new_data


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

        model = train_model(
            train_data,
            embeddings,
            pad_lens,
            num_words=FLAGS.num_words,
            model_dir=FLAGS.model_dir,
            model_name=FLAGS.model_name,
            fig_name='history',
            epochs=FLAGS.epochs,
            net_scale=FLAGS.net_scale,
        )
    elif FLAGS.mode == 'train_k_fold':
        train_df = pd.read_csv('./tmp/train.csv',
                               converters={"hashtags": literal_eval})
        test_df = pd.read_csv('./tmp/test.csv')
        train_data, test_data, embeddings, pad_lens = prepare_data(
            train_df, test_df, num_words=FLAGS.num_words)

        train_data = subset_data(train_data, FLAGS.subset)
        model = train_k_folds(train_data,
                              embeddings,
                              pad_lens,
                              num_words=FLAGS.num_words,
                              model_dir=FLAGS.model_dir,
                              model_name=FLAGS.model_name,
                              fig_name='history',
                              epochs=FLAGS.epochs,
                              net_scale=FLAGS.net_scale)
    elif FLAGS.mode == 'gen_submission':
        train_df = pd.read_csv('./tmp/train.csv',
                               converters={"hashtags": literal_eval})
        test_df = pd.read_csv('./tmp/test.csv')
        train_data, test_data, embeddings, pad_lens = prepare_data(
            train_df, test_df, num_words=FLAGS.num_words)
        gen_submission(test_data,
                       model_dir=FLAGS.model_dir,
                       model_name=FLAGS.model_name)
    else:
        raise Exception('improper selection for mode')

    return


if __name__ == "__main__":
    app.run(main)