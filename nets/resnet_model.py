from nets.components.bidir_text_net import bidir_text_net
from nets.components.text_net import text_net
from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def resnet_model(embedding_matrix,
                 max_lens: Dict,
                 tokenizer_len: int = 100,
                 net_scale: int = 64,
                 fig_path: str = './model.png') -> tf.keras.Model:

    ((input_text, input_loc, input_ky, input_hashtag),
     merge) = bidir_text_net(embedding_matrix, max_lens, tokenizer_len,
                             net_scale)

    dropout = layers.Dropout(0.5)(merge)
    dense1 = layers.Dense(1024, activation='relu')(dropout)
    res1 = layers.concatenate([merge, dense1])

    dropout2 = layers.Dropout(0.5)(res1)
    dense2 = layers.Dense(512, activation='relu')(dropout2)
    res2 = layers.concatenate([res1, dense2])

    final_dense = layers.Dense(2)(res2)
    output = layers.Activation('softmax')(final_dense)

    model = tf.keras.Model(inputs={
        'text': input_text,
        'keyword': input_ky,
        'location': input_loc,
        'hashtags': input_hashtag
    },
                           outputs=output)
    tf.keras.utils.plot_model(model, to_file=fig_path, show_shapes=True)
    return model