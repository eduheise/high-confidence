from .float_metric import FloatMean, FloatMetric

class FloatConfusionMatrix(FloatMetric):
    def __init__(self, decay):
        self.tp = FloatMean(decay)
        self.fp = FloatMean(decay)
        self.tn = FloatMean(decay)
        self.fn = FloatMean(decay)
        super().__init__(decay)
    
    def update(self, value, label):
        new_tp = 1 if label and value else 0
        new_fp = 1 if not label and value else 0
        new_tn = 1 if not label and not value else 0
        new_fn = 1 if label and not value else 0
        
        self.tp.update(new_tp)
        self.fp.update(new_fp)
        self.tn.update(new_tn)
        self.fn.update(new_fn)

    def tp(self):
        return self.tp.get()
    
    def fp(self):
        return self.fp.get()
    
    def tn(self):
        return self.tn.get()
    
    def fn(self):
        return self.fn.get()