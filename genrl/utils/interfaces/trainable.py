from abc import abstractmethod
from typing import Dict, Protocol


class Trainable(Protocol):
    @abstractmethod
    def build(self):
        raise NotImplementedError()

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict:
        """
            Update the model parameter
            :return log dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict:
        """
        Evaluate the skill decoder (e.g, MSE with true action, ...)
        """
        raise NotImplementedError()
