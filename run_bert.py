import os

import pandas as pd
from tensorflow.keras.utils import plot_model

from nets.bert_model import build_bert_model
from preprocessing.bert_preprocess import bert_preprocess


def run_bert(train_df: pd.DataFrame,
             test_df: pd.DataFrame,
             model_dir: str = './tmp/bert',
             model_name: str = 'bert_default'):
    os.makedirs(model_dir, exist_ok=True)
    preprocess_data = bert_preprocess(train_df, test_df)
    bert_layer = preprocess_data['bert_layer']
    max_token_len = preprocess_data['max_token_len']
    model = build_bert_model(bert_layer, max_len=max_token_len)()
    plot_model(model,
               to_file=os.path.join(model_dir, model_name + '.png'),
               show_shapes=True)
    return
