import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from typing import Dict, Sequence, Tuple
from ast import literal_eval

import pandas as pd

from absl import app, flags, logging

from training import train_model, train_k_folds
from preprocessing import preprocess, prepare_data
from gen_submission import gen_submission
from nets.avg_ensemble import avg_ensemble
from training.train_k_folds import train_k_folds
from preprocessing.preprocess_bert import preprocess_bert
from training.train_bert import train_bert
from nets.bert.bert_ensemble import create_bert_ensemble

logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS

flags.DEFINE_enum('mode', 'gen_submission', [
    'preprocess', 'preprocess_bert', 'train', 'train_k_fold', 'train_bert',
    'gen_submission', 'create_ensemble', 'create_bert_ensemble'
], 'defines mode to run app')
flags.DEFINE_string('train_path', './data/train.csv', 'path of train.csv')
flags.DEFINE_string('test_path', './data/test.csv', 'path of train.csv')
flags.DEFINE_integer('num_words', 35000, 'number of words to use')
flags.DEFINE_string('model_dir', './tmp/model/',
                    'directory to which to save model')
flags.DEFINE_string('model_name', 'model', 'model name')
flags.DEFINE_integer('epochs', 25, 'number of training epochs')
flags.DEFINE_integer('net_scale', 64, 'scaling for network sizes')
flags.DEFINE_float('subset', 1.0, 'subset ratio of data')
flags.DEFINE_float('learn_rate', 0.001, 'learning rate')


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
    elif FLAGS.mode == 'preprocess_bert':
        train_df = pd.read_csv('./tmp/train.csv',
                               converters={"hashtags": literal_eval})
        test_df = pd.read_csv('./tmp/test.csv')
        preprocess_bert(train_df, test_df)
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
                              net_scale=FLAGS.net_scale,
                              learn_rate=FLAGS.learn_rate)
    elif FLAGS.mode == 'train_bert':
        train_bert(model_dir=FLAGS.model_dir,
                   model_name=FLAGS.model_name,
                   epochs=FLAGS.epochs,
                   subset=FLAGS.subset)
    elif FLAGS.mode == 'create_ensemble':
        avg_ensemble(FLAGS.model_dir)
    elif FLAGS.mode == 'create_bert_ensemble':
        create_bert_ensemble(model_dir=FLAGS.model_dir,
                             model_name=FLAGS.model_name)
    elif FLAGS.mode == 'gen_submission':
        train_df = pd.read_csv('./tmp/train.csv',
                               converters={"hashtags": literal_eval})
        test_df = pd.read_csv('./tmp/test.csv')
        train_data, test_data, embeddings, pad_lens = prepare_data(
            train_df, test_df, num_words=FLAGS.num_words)
        gen_submission(train_data,
                       test_data,
                       model_dir=FLAGS.model_dir,
                       model_name=FLAGS.model_name)
    else:
        raise Exception('improper selection for mode')

    return


if __name__ == "__main__":
    app.run(main)