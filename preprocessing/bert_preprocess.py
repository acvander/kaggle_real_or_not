import numpy as np
import pandas as pd
import tensorflow_hub as hub

import spacy

from nets.bert.bert_tokenizer import FullTokenizer


def bert_encode(texts, tokenizer, max_len=512):
    '''taken from lhideki/bert-with-kfold'''
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[:max_len - 2]
        input_sequence = ['[CLS]'] + text + ['[SEP]']
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return {
        'tokens': np.array(all_tokens),
        'masks': np.array(all_masks),
        'segments': np.array(all_segments)
    }


def bert_preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    nlp = spacy.load('en_core_web_sm')

    train_df['text_len'] = train_df['text'].apply(lambda x: len(nlp(x)))
    test_df['text_len'] = test_df['text'].apply(lambda x: len(nlp(x)))
    max_token_len = max(train_df['text_len'].max(),
                        test_df['text_len'].max()) + 2
    bert_layer = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1",
        trainable=True)

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

    tokenizer = FullTokenizer(vocab_file, do_lower_case)

    train_input = bert_encode(train_df['text'].values,
                              tokenizer,
                              max_len=max_token_len)
    train_output = train_df['target'].to_numpy()

    test_input = bert_encode(test_df['text'].values,
                             tokenizer,
                             max_len=max_token_len)

    return {
        'train_input': train_input,
        'train_output': train_output,
        'test_input': test_input,
        'bert_layer': bert_layer,
        'max_token_len': max_token_len
    }
