from .float_utils import FloatSTD, FloatMean
from .base import AdaptiveThresholdMixin
import logging
import math
import numpy as np

class LogisticADADRIFT(AdaptiveThresholdMixin):
    """Logistic ADADRIFT

    Logistic ADADRIFT is an implementation of ADADRIFT using logistic and hyperbolic tangent activate function (instead of the standard 
    exponential function).

    ADADRIFT reference:
    E. Ferreira José, F. Enembreck and J. Paul Barddal, "ADADRIFT: An Adaptive Learning Technique for Long-history 
    Stream-based Recommender Systems," 2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC), 
    2020, pp. 2593-2600, doi: 10.1109/SMC42975.2020.9282922.

    """
    def __init__(self, s_window=1000, confidence=.985, deviation=.001, max_prob=0.99):
        """Initialization function

        Args:
            s_window (int, optional): Size of the sliding window. Defaults to 1000.
            confidence (float, optional): Fixed confidence that works like the long-term moving average. Defaults to .985
            deviation (float, optional): Fixed deviation that works like the long-term moving standard deviation. Defaults to .001
        """

        self.default_window = s_window
        self.default_confidence = confidence
        self.default_deviation = deviation
        self.max_prob = max_prob

        self.confidence = {}
        self.deviation = {}
        self.s_decay = {} 
        self.short_term = {}
        self.gamma = {}

    def is_auto(self, id, prob):
        """Is automatic?

        Args:
            id (str): model ID (i.e. application01)
            prob (float): probability

        Returns:
            int: representing if this probability should be an automatic output
        """
        if self.get_performance(id) > 0.98 and prob > self.threshold(id):
            return 1
        else:
            return 0

    def register(self, id):
        """Register a new id

        Args:
            id (str): Model ID (i.e. application01)
        """
        if not id in self.short_term:
            self.confidence[id] = self.default_confidence
            self.deviation[id] = self.default_deviation
            self.s_decay[id] = (self.default_window - 1) / self.default_window 

            self.short_term[id] = FloatMean(self.s_decay[id])
            self.gamma[id] = self.confidence[id]

    def threshold(self, id):
        """Get threshold of a model

        Args:
            id (str): Model ID (i.e. application01)

        Returns:
            float: Calculates the actual threshold
        """

        prob_rate = (self.max_prob - 0.5) * 2

        th = self.sigmoid(self.gamma[id]) * prob_rate
        th = th / 2 + .5
        return th

    def get_performance(self, id):
        """Get performance of a model

        Args:
            id (str): Model ID (i.e. application01)

        Returns:
            float: Accuracy representing the performance of the threshold
        """
        return self.short_term[id].get()

    def _update_threshold(self, id, change):
        """Update threshold private method

        Args:
            id (str): Model ID (i.e. application01)
            change (float): The difference between the actual performance and the long-term confidence
        """
        self.gamma[id] = self.gamma[id] * self.s_decay[id] + 10 * np.tanh(change) * (1 - self.s_decay[id])

    def update(self, id, auto, label):
        """Update ADADRIFT with a new instance

        Args:
            id (str): Model ID (i.e. application01)
            auto (bool): If this instance was automatically released
            label (int): Is the label correct classified?
        """
        if auto:
            self.short_term[id].next(label)
            diff = self.confidence[id]  - self.short_term[id].get()
            change = (diff/self.deviation[id])

            self._update_threshold(id, change)
    
    def sigmoid(self, x):
        """Simple sigmoid
        """
        return 1 / (1 + math.exp(-x))

    def deploy_it(self, id):
        """Output used to dump in the threshold.json file

        Args:
            id (str): Model ID (i.e. application01)

        Returns:
            dict: Field for each model in the threshold.json file
        """
        return {'is_auto': int(self.get_performance(id) > 0.98),
                'threshold': self.threshold(id),
                'performance': self.get_performance(id)}

    def set_confidence(self, model_id, confidence):
        self.confidence[model_id] = confidence

    def set_deviation(self, model_id, deviation):
        self.deviation[model_id] = deviation

    def set_window(self, model_id, window):
        self.s_decay[model_id] = (window - 1) / window 
        self.short_term[model_id].decay = self.s_decay[model_id]
        
