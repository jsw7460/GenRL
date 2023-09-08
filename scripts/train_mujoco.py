import sys

sys.path.append("/home/jsw7460/diffusion_rl/")

from typing import Dict

import hydra
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

from genrl.rl.buffers.genrl_buffer import GenRLDataset
from genrl.algos import BC


@hydra.main(version_base=None, config_path="../config/train", config_name="base")
def program(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)  # type: Dict

    buffer = GenRLDataset(cfg["env"]["dataset_path"], 777)

    # buffer = buffer.filter_episodes(lambda ep: ep.total_timesteps > 30)
    train_dt, eval_dt = GenRLDataset.split_dataset(buffer, sizes=[len(buffer) - 3, 3])
    train_dt.cache_data()
    print("Train dataset", len(train_dt))
    # print("Eval dataset", len(eval_dt))

    modules_dict = {}
    for module in cfg["algo"]["modules"]:
        target = get_class(cfg[module]["target"])
        modules_dict[module] = target(**cfg[module]["kwargs"])

    algo_cls = get_class(cfg["algo"]["cls"])
    algo = algo_cls(cfg=cfg, env=None, **modules_dict)  # type: BC

    algo.train_dataset = train_dt
    algo.eval_dataset = eval_dt

    algo.train()


if __name__ == "__main__":
    program()
