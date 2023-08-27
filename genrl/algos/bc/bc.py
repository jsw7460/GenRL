from typing import Dict

from genrl.algos.base import BaseTrainer
from genrl.policies.low.base import BaseLowPolicy
from genrl.utils.common.type_aliases import GymEnv
from genrl.rl.buffers.genrl_buffer import GenRLDataset
from genrl.rl.buffers.type_aliases import GenRLBufferSample


class BC(BaseTrainer):

    def __init__(self, cfg: Dict, env: GymEnv, low_policy: BaseLowPolicy):
        """
        :param cfg:
        :param env:
        :param low_policy:
        This class is not responsible for filling the replay buffer.

        """
        super(BC, self).__init__(cfg=cfg, env=env)
        self.low_policy = low_policy
        self._train_dataset = None
        self._eval_dataset = None

    @property
    def train_dataset(self) -> GenRLDataset:
        return self._train_dataset

    @property
    def eval_dataset(self) -> GenRLDataset:
        return self._eval_dataset

    @train_dataset.setter
    def train_dataset(self, dataset):
        assert len(dataset) > 0, "Dataset should have more than one trajectory"
        self._train_dataset = dataset

    @eval_dataset.setter
    def eval_dataset(self, dataset):
        assert len(dataset) > 0, "Dataset should have more than one trajectory"
        self._eval_dataset = dataset

    def _prepare_run(self) -> None:
        self.required_total_update = self.max_iter

    def _sample_train_batch(self) -> GenRLBufferSample:
        return self.train_dataset.sample_subtrajectories(
            self.batch_size,
            self.subseq_len,
            allow_replace=True
        )

    def _update_model(self, replay_data: GenRLBufferSample) -> Dict:
        info = self.low_policy.update(replay_data)
        return info
