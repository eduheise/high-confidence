from collections import deque
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging

class ContinuousReevaluation:

    """Base class for experimentation of the adaptive threshold methods.
    """

    def __init__(self, streamer, models=[], delay_s=3600):
        """Set/initialize parameters.
        Args:
            streamer (Streamer): Instance of a streamer which has been initialized.
            models (list[str]): Models bein evaluated
            delay_s (int): Delay in seconds to update each model.
        """
        self.streamer = streamer
        self.models = models
        self.delay = delay_s * 1000

    def fit(self, events, timestamp_col='timestamp'):
        """Models are incrementally updated using events.

        Args:
            events (pd.DataFrame): Events returned by the BaseC4i0DB.
            timestamp_col (str, optional): Column representing the timestamp. Defaults to 'timestamp'.
        """
        print("Fit validation started..")
        try:
            events = events.sort_values(by=timestamp_col)
        except KeyError as e:
            raise KeyError('É necessário que seja sinalizado o nome do campo do timestamp (padrão é timestamp).')

        # make initial status for batch training
        self._validate()

        print("Updating model with training instances..")
        for e in tqdm(events.iterrows(), total=events.shape[0]):
            self._update([e[1]])

    def evaluate(self, events, timestamp_col='timestamp'):
        """Models are incrementally updated using events.

        Args:
            events (pd.DataFrame): Events returned by the BaseC4i0DB.
            timestamp_col (str, optional): Column representing the timestamp. Defaults to 'timestamp'.
        """
        try:
            events = events.sort_values(by=timestamp_col)
        except KeyError as e:
            raise KeyError('É necessário que seja sinalizado o nome do campo do timestamp (padrão é timestamp).')

 
        responses = []
        print("Prequential evaluation started...")
        last_eval = -np.inf
        train_queue = []
        for e in tqdm(events.iterrows(), total=events.shape[0]):
            response = self._evaluate(e[1])
            response['timestamp'] = e[1]['timestamp']
            responses.append(response)
            train_queue.append(e[1])
            if e[1]['timestamp'] > last_eval + self.delay:
                self._update(train_queue)
                train_queue = []
                last_eval = e[1]['timestamp']
        return pd.DataFrame(responses)

    def _validate(self):
        print(f"Validating models {self.models}")
        
        for model in self.models:
            self.streamer.register(model)
    
    def _update(self, events):
        for model in self.models:
            ths = self.streamer.threshold(model)
            for e in events:
                auto = e[f'prob_{model}'] > ths
                self.streamer.update(model, auto, e[f'label_{model}'])

    def _evaluate(self, e):
        response = {}
        for model in self.models:
            threshold = self.streamer.threshold(model)
            err = self.streamer.get_performance(model)
            response[f'auto_{model}'] = self.streamer.is_auto(model, e[f'prob_{model}'])
            response[f'label_{model}'] = e[f'label_{model}']
            response[f'prob_{model}'] = e[f'prob_{model}']
            response[f'err_{model}'] = err
            response[f'ths_{model}'] = threshold
            
        return response
