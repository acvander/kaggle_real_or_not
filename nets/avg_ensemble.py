from pathlib import Path
import os

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.utils import plot_model


def avg_ensemble(model_dir: str):
    sub_models = []
    model_files = Path(model_dir).glob('*.h5')
    for model in model_files:
        # skip ensemble model
        if model.stem == 'ensemble':
            continue
        sub_models.append(load_model(model, compile=False))
    text_input = layers.Input(shape=(sub_models[0].input['text'].shape[1:]),
                              name='text')
    kw_input = layers.Input(shape=(sub_models[0].input['keyword'].shape[1:]),
                            name='keyword')
    loc_input = layers.Input(shape=(sub_models[0].input['location'].shape[1:]),
                             name='location')
    hashtags_input = layers.Input(
        shape=(sub_models[0].input['hashtags'].shape[1:]), name='hashtags')
    inputs = {
        'text': text_input,
        'keyword': kw_input,
        'location': loc_input,
        'hashtags': hashtags_input
    }
    model_outputs = []
    for model in sub_models:
        # remove last layer which should be the activation layer
        model.layers.pop(-1)
        output = model.layers[-1].output
        model = Model(inputs=model.input, outputs=output)
        model_outputs.append(model(inputs))
    average = layers.average([model for model in model_outputs])
    softmax = layers.Activation(tf.nn.softmax)(average)
    ensemble = Model(inputs=inputs, outputs=softmax)
    save_model(ensemble, os.path.join(model_dir, 'ensemble.h5'))
    plot_model(ensemble,
               to_file=os.path.join(model_dir, 'ensemble.png'),
               show_shapes=True)
    return ensemble
