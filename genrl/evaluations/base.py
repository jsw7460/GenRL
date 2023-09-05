import pickle
import random
from pathlib import Path
from typing import Dict, Union, Callable, Type, Tuple

import numpy as np
from hydra.utils import get_class
from omegaconf import DictConfig

from genrl.rl.envs import GenRLVecEnv, GenRLHistoryEnv
from genrl.utils.common.dump import dump_video, dump_text
from genrl.utils.common.type_aliases import GenRLEnvEvalResult
from genrl.utils.interfaces import JaxSavable


class EvaluationExecutor:
    save_filename_base = "{env}_r{reward}"
    max_env_name = 15

    def __init__(self, cfg: Union[Dict, DictConfig], envs: Tuple[GenRLHistoryEnv, ...]):
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.cfg = cfg
        with open(cfg.pretrained_path, "rb") as f:
            self.pretrained_cfg = pickle.load(f)

        self.subseq_len = self.pretrained_cfg["subseq_len"]
        self.render = cfg["render"]
        self.visual_save_path = None
        self.text_save_path = None

        date = self.pretrained_cfg["date"]
        model_suffix = self.cfg["pretrained_suffix"]
        pretrained_step = str(self.cfg["step"])

        self.visual_save_path \
            = Path(self.cfg["visual_save_prefix"]) / Path(date) / Path(model_suffix + "_" + pretrained_step)
        self.text_save_path \
            = Path(self.cfg["text_save_prefix"]) / Path(date) / Path(model_suffix + "_" + pretrained_step)

        self.text_save_path.mkdir(parents=True, exist_ok=True)
        if self.render:
            self.visual_save_path.mkdir(parents=True, exist_ok=True)

        [env.setup(self.subseq_len) for env in envs]
        self.vectorized_env = GenRLVecEnv(envs)
        self.pretrained_models = {}
        self._load_models()

    def _load_models(self) -> None:
        for module in self.pretrained_cfg["modules"]:
            cls = get_class(self.pretrained_cfg[module]["target"])  # type: Union[type, Type[JaxSavable]]
            instance = cls.load(f"{self.pretrained_cfg['save_paths'][module]}_{self.cfg.step}")
            self.pretrained_models[module] = instance

    def eval_execute(self, eval_fn: Callable[..., GenRLEnvEvalResult], **kwargs):
        predictable_module = self.pretrained_cfg["algo"]["modules"][0]
        models = {"model": self.pretrained_models[predictable_module]}
        eval_result = eval_fn(**models, env=self.vectorized_env, render=self.render, **kwargs)
        rewards = eval_result.episode_rewards
        episode_lengths = eval_result.episode_lengths
        vis_observations = eval_result.vis_observations

        text_to_dump = "=" * 50
        for t, env in enumerate(self.vectorized_env):
            reward = round(rewards[t], 2)
            ep_len = episode_lengths[t]
            filename_base = self.save_filename_base.format(env=repr(env), reward=reward)

            text_to_dump += f"\nEnv: {repr(env):<{self.max_env_name}} Reward mean: {reward}, Episode len: {ep_len}"

            if self.render:
                video = vis_observations[t]
                video_path = self.visual_save_path / Path(filename_base)
                dump_video(video=video, path=str(video_path))

        text_to_dump += "\n" + "=" * 50
        dump_text(text=text_to_dump, path=str(self.text_save_path / Path("eval.txt")))
        print(text_to_dump + "\n")
