class AdaptiveThresholdMixin:
    def threshold(self, id):
        """Get threshold method

        Args:
            id (string): An unique ID representing the adaptive threshold instance

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def update(self, id, prob, label):
        """Update threshold

        Args:
            id (string): An unique ID representing the adaptive threshold instance
            prob (float): Probability given by the model in this instance
            label (int): The instance's label 

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError
