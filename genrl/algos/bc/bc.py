from typing import Dict

from genrl.algos.base import BaseTrainer
from genrl.policies.low.base import BaseLowPolicy
from genrl.utils.common.type_aliases import GymEnv
from genrl.rl.buffers.genrl_buffer import GenRLDataset


class BC(BaseTrainer):

    def __init__(self, cfg: Dict, env: GymEnv, low_policy: BaseLowPolicy):
        """
        :param cfg:
        :param env:
        :param policy:
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

    def _prepare_run(self):
        self.required_total_update = self.max_iter

    def learn(self):
        for update_step in range(self.required_total_update):
            replay_data = self.train_dataset.sample_subtrajectories(
                n_episodes=self.batch_size,
                subseq_len=self.subseq_len
            )
            info = self.low_policy.update(replay_data)

            # print(info[])
