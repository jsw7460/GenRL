from typing import Protocol

from genrl.utils.common.type_aliases import GenRLPolicyInput, GenRLPolicyOutput


class PolicyPredictor(Protocol):
    def predict(self, x: GenRLPolicyInput) -> GenRLPolicyOutput:
        """
        Model for prediction (a protocol used for the environment evaluation)
        :param x:
        :return:
        """
