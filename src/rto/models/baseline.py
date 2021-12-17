from .float_utils import f1_score
from .base import AdaptiveThresholdMixin

class ConstantThreshold(AdaptiveThresholdMixin):
    """Work still in progress
    """
    def __init__(self, confidence=.95):
        self.confidence = confidence

    def threshold(self, *unused):
        return self.confidence

    def update(self, *unused):
        return
