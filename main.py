from metrics.CustomF1 import CustomF1Score
from typing import Dict, Sequence, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers

from tensorflow_addons.metrics import FBetaScore, F1Score

TRAIN_PATH = './data/train.csv'
TEST_PATH = './data/test.csv'
NUM_WORDS = 35000
MODEL_PATH = './tmp/model.h5'


def clean_row(row: pd.Series):
    # remove %20
    row['keyword'] = row['keyword'].replace('%20', ' ')
    return row


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # replace NaNs with empty string
    df = df.fillna(value='')
    clean_df = df.apply(clean_row, axis=1)
    return clean_df


def prepare_data(train_df: pd.DataFrame,
                 test_df: pd.DataFrame) -> Tuple[Dict, Dict, Tokenizer, Dict]:
    text_lengths = [len(x.split()) for x in train_df['text'].to_list()]
    kw_lengths = [len(x.split()) for x in train_df['keyword'].to_list()]
    loc_lengths = [len(x.split()) for x in train_df['location'].to_list()]

    max_text_len = max(text_lengths) + 1
    max_kw_len = max(kw_lengths) + 1
    max_loc_len = max(loc_lengths) + 1

    # tokenize data
    tokenizer = Tokenizer(num_words=NUM_WORDS, lower=True)
    tokenizer.fit_on_texts((train_df['keyword'] + train_df['location'] +
                            train_df['text'].to_list()))

    train_text = tokenizer.texts_to_sequences(train_df['text'])
    train_keyword = tokenizer.texts_to_sequences(train_df['keyword'])
    train_location = tokenizer.texts_to_sequences(train_df['location'])

    test_text = tokenizer.texts_to_sequences(test_df['text'])
    test_keyword = tokenizer.texts_to_sequences(test_df['keyword'])
    test_location = tokenizer.texts_to_sequences(test_df['location'])

    label = train_df['target'].to_list()
    label = tf.keras.utils.to_categorical(label)

    # pad sequences
    train_text = pad_sequences(train_text, maxlen=max_text_len)
    train_keyword = pad_sequences(train_keyword, maxlen=max_kw_len)
    train_location = pad_sequences(train_location, maxlen=max_loc_len)

    test_text = pad_sequences(test_text, maxlen=max_text_len)
    test_keyword = pad_sequences(test_keyword, maxlen=max_kw_len)
    test_location = pad_sequences(test_location, maxlen=max_loc_len)

    train_data = {
        'text': train_text,
        'keyword': train_keyword,
        'location': train_location,
        'target': label
    }
    test_data = {
        'text': test_text,
        'keyword': test_keyword,
        'location': test_location,
    }
    max_lens = {
        'text': max_text_len,
        'keyword': max_kw_len,
        'location': max_loc_len
    }
    return train_data, test_data, tokenizer, max_lens


def load_embeddings(
        tokenizer: tf.keras.preprocessing.text.Tokenizer,
        embeddings_file: str = './data/glove.6B.100d.txt',
        embedding_dim: int = 100,
        max_words: int = 1000,
) -> np.array:
    embedding_dict = {}
    with open(embeddings_file) as glove_file:
        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = np.asarray(records[1:], dtype='float32')
            embedding_dict[word] = vector_dimensions

    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, idx in tokenizer.word_index.items():
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector

    return embedding_matrix


def build_model(embedding_matrix, max_lens: Dict, tokenizer_len: int = 100):
    input_text = layers.Input(shape=(max_lens['text'], ), name='text')
    embedding_text = layers.Embedding(tokenizer_len,
                                      100,
                                      weights=[embedding_matrix],
                                      input_length=max_lens['text'],
                                      trainable=False)(input_text)
    dropout_text = layers.Dropout(0.2)(embedding_text)
    lstm_text_1 = layers.LSTM(128, return_sequences=True)(dropout_text)
    lstm_text_2 = layers.LSTM(128, return_sequences=True)(lstm_text_1)
    lstm_text_3 = layers.LSTM(128, return_sequences=True)(lstm_text_2)
    lstm_text_4 = layers.LSTM(128)(lstm_text_3)

    input_ky = layers.Input(shape=(max_lens['keyword'], ), name='keyword')
    embedding_ky = layers.Embedding(tokenizer_len,
                                    100,
                                    weights=[embedding_matrix],
                                    input_length=max_lens['keyword'],
                                    trainable=False)(input_ky)
    dropout_ky = layers.Dropout(0.2)(embedding_ky)
    lstm_ky_1 = layers.LSTM(64, return_sequences=True)(dropout_ky)
    lstm_ky_2 = layers.LSTM(64, return_sequences=True)(lstm_ky_1)
    lstm_ky_3 = layers.LSTM(64, return_sequences=True)(lstm_ky_2)
    lstm_ky_4 = layers.LSTM(64)(lstm_ky_3)

    input_loc = layers.Input(shape=(max_lens['location'], ), name='location')
    embedding_loc = layers.Embedding(tokenizer_len,
                                     100,
                                     weights=[embedding_matrix],
                                     input_length=max_lens['location'],
                                     trainable=False)(input_loc)
    dropout_loc = layers.Dropout(0.2)(embedding_loc)
    lstm_loc_1 = layers.LSTM(32, return_sequences=True)(dropout_loc)
    lstm_loc_2 = layers.LSTM(32, return_sequences=True)(lstm_loc_1)
    lstm_loc_3 = layers.LSTM(32, return_sequences=True)(lstm_loc_2)
    lstm_loc_4 = layers.LSTM(32)(lstm_loc_3)

    merge = layers.concatenate([lstm_text_4, lstm_ky_4, lstm_loc_4])

    dropout = layers.Dropout(0.5)(merge)
    dense1 = layers.Dense(256, activation='relu')(dropout)
    dense2 = layers.Dense(128, activation='relu')(dense1)
    output = layers.Dense(2, activation='softmax')(dense2)

    model = tf.keras.Model(inputs={
        'text': input_text,
        'keyword': input_ky,
        'location': input_loc
    },
                           outputs=output)
    tf.keras.utils.plot_model(model,
                              to_file="./tmp/model.png",
                              show_shapes=True)
    return model


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    columns = train_df.columns.to_list()
    example = train_df.iloc[100].to_list()

    clean_train_df = clean_data(train_df)
    clean_test_df = clean_data(test_df)

    train_data, test_data, tokenizer, pad_lens = prepare_data(
        clean_train_df, clean_test_df)
    embeddings = load_embeddings(tokenizer, max_words=NUM_WORDS)

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