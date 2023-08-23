from abc import ABCMeta, abstractmethod
from typing import Dict

import numpy as np

from genrl.utils.jax_utils.model import Model


class ITrainable(metaclass=ABCMeta):
	@abstractmethod
	def build(self):
		raise NotImplementedError()

	@abstractmethod
	def predict(self, *args, **kwargs) -> np.ndarray:
		"""Forward of skill decoder model"""
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
