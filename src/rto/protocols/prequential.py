from collections import deque
from tqdm import tqdm
import pandas as pd
import logging

class Prequential:

    """Base class for experimentation of the adaptive threshold methods.
    """

    def __init__(self, streamer, models=[]):
        """Set/initialize parameters.
        Args:
            streamer (Streamer): Instance of a streamer which has been initialized.
            repeat (boolean): Choose whether the same item can be repeatedly interacted by the same user.
            maxlen (int): Size of an item buffer which stores most recently observed items.
        """
        self.streamer = streamer
        self.models = models

    def fit(self, events, timestamp_col='timestamp'):
        """Train a model using the first 30% positive events to avoid cold-start.
        Evaluation of this batch training is done by using the next 20% positive events.
        After the batch SGD training, the models are incrementally updated by using the 20% test events.
        Args:
            events (list of Event): Positive training events (0-30%).
            train_size (float [0,1]): Train sample.
            models (list of columns): Columns to be evaluated
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
            self._update(e[1])

    def evaluate(self, events, timestamp_col='timestamp'):
        """Iterate recommend/update procedure and compute incremental recall.
        Args:
            test_events (list of Event): Positive test events.
        Returns:
            list of tuples: (score recall@1, rank, recommend time, update time)
        """
        try:
            events = events.sort_values(by=timestamp_col)
        except KeyError as e:
            raise KeyError('É necessário que seja sinalizado o nome do campo do timestamp (padrão é timestamp).')

 
        responses = []
        print("Prequential evaluation started...")
        for e in tqdm(events.iterrows(), total=events.shape[0]):
            
            response = self._evaluate(e[1])
            responses.append(response)

            self._update(e[1])

            # (where the correct item is ranked, correct rating, predicted rating, user index, rec time, update time)
        return pd.DataFrame(responses)

    def _validate(self):
        print(f"Validating models {self.models}")
        
        for model in self.models:
            self.streamer.register(model)
    
    def _update(self, e):
        for model in self.models:
            auto = e[f'prob_{model}'] > self.streamer.threshold(model)
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
