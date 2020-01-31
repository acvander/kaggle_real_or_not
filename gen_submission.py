import os
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_addons.metrics import F1Score


def gen_submission(test_data: Dict,
                   model_dir: str = './tmp/',
                   model_name: str = 'model'):
    submission_path = os.path.join(model_dir, 'submission.csv')
    model_path = os.path.join(model_dir, model_name) + '.h5'
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={'f1_score': F1Score},
                                       compile=False)

    results = model.predict(test_data, batch_size=128, verbose=1)

    # pick best result from each pair
    target = np.argmax(results, axis=1)

    # combine ids and target into dataframe
    df = pd.DataFrame(data={'id': test_data['id'], 'target': target})

    df.to_csv(submission_path, index=False)
    return