from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_addons.metrics import F1Score


def gen_submission(test_data: Dict, model_path: str = './tmp/model.h5'):
    submission_path = './tmp/submission.csv'
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