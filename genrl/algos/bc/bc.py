from typing import Dict, Any

import numpy as np

from genrl.algos.base import BaseTrainer
from genrl.policies.base import BasePolicy
from genrl.rl.buffers.genrl_buffer import GenRLDataset
from genrl.rl.buffers.type_aliases import GenRLBufferSample
from genrl.utils.common.type_aliases import GymEnv, GenRLPolicyInput


class BC(BaseTrainer):

    def __init__(self, cfg: Dict, env: GymEnv, low_policy: BasePolicy):
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
        return self.train_dataset.sample_subtrajectories(n_episodes=self.batch_size, subseq_len=self.subseq_len)

    def _update_model(self, replay_data: GenRLBufferSample) -> Dict:
        info = self.low_policy.update(replay_data)
        return info

    def _evaluate_model(self) -> Dict[str, Any]:
        subtraj = self.eval_dataset.sample_subtrajectories(n_episodes=self.batch_size, subseq_len=self.subseq_len)

        policy_input = GenRLPolicyInput(
            observations=subtraj.observations,
            actions=subtraj.actions,
            masks=subtraj.masks,
            rewards=subtraj.rewards,
            terminations=subtraj.terminations,
            timesteps=subtraj.timesteps_range
        )

        policy_output = self.low_policy.predict(policy_input)

        pred_action = policy_output.pred_action
        policy_info = policy_output.info

        targ_action = subtraj.actions
        action_dim = targ_action.shape[-1]

        pred_action = pred_action.reshape(-1, action_dim) * subtraj.masks.reshape(-1, 1)
        targ_action = targ_action.reshape(-1, action_dim) * subtraj.masks.reshape(-1, 1)

        mse_loss = np.mean((pred_action - targ_action) ** 2, axis=-1)
        mse_loss = np.sum(mse_loss) / np.sum(subtraj.masks)

        policy_info.update({"mse_loss": mse_loss})

        return policy_info
