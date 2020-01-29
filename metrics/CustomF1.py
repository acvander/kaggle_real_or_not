from tensorflow.keras.metrics import Metric, Recall, Precision


class CustomF1Score(Metric):
    def __init__(self, name=None, **kwargs):
        super(Metric, self).__init__(name=name, **kwargs)
        self.recall = Recall()
        self.precision = Precision()
        self.score = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.recall.update_state(y_true, y_pred, sample_weight=sample_weight)
        self.precision.update_state(y_true,
                                    y_pred,
                                    sample_weight=sample_weight)

        recall = self.recall.result()
        precision = self.precision.result()

        f1score = 2 * precision * recall / (precision + recall)
        self.score = f1score

    def result(self):
        return self.score

    def reset_states(self):
        self.recall.reset_states()
        self.precision.reset_states()
        self.score = 0