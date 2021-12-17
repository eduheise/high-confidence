from .float_metric import FloatMetric, FloatMean
from .float_confusion_matrix import FloatConfusionMatrix

class FloatF1(FloatMetric):
    def __init__(self, decay):
        self.confusion_matrix = FloatMatrixConfusion(decay)
        super().__init__(decay)

    def update(self, value, label):
        self.confusion_matrix.update(value, label)

    def get(self):
        tp = self.confusion_matrix.tp()
        fn = self.confusion_matrix.fn()
        fp = self.confusion_matrix.fp()

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)

        return 2 * ((precision * recall) / (precision + recall))


