from typing import Dict, Any

import numpy as np

from genrl.algos import BC
from genrl.policies.base import BasePolicy
from genrl.utils.common.type_aliases import GymEnv, GenRLPolicyInput


class SkillBC(BC):

    def __init__(self, cfg: Dict, env: GymEnv, skill: BasePolicy):
        """
        :param cfg:
        :param env:
        :param skill:
        This class is not responsible for filling the replay buffer.
        Skill is same as low policy, but conditioned on some information (semantic skill).

        """
        super(SkillBC, self).__init__(cfg=cfg, env=env, low_policy=skill)

        self.skill = self.low_policy

    def _evaluate_model(self) -> Dict[str, Any]:
        subtraj = self.eval_dataset.sample_subtrajectories(n_episodes=self.batch_size, subseq_len=self.subseq_len)

        policy_input = GenRLPolicyInput(
            observations=subtraj.observations,
            actions=subtraj.actions,
            sem_skills=subtraj.sem_skills,
            sem_skills_done=subtraj.sem_skills_done,
            masks=subtraj.masks,
            rewards=subtraj.rewards,
            terminations=subtraj.terminations,
            timesteps=subtraj.timesteps_range
        )

        policy_output = self.skill.predict(policy_input)

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
