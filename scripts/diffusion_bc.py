import sys
sys.path.append("/home/jsw7460/diffusion_rl/")

from typing import Dict

import h5py
import hydra
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf

from genrl.rl.buffers.genrl_buffer import GenRLDataset
from genrl.algos import BC

def apply_fn(ep_dict):
    dict_obs = ep_dict["observations"]
    np_obs = dict_obs["observation"]
    ep_dict.update({"observations": np_obs})
    return ep_dict

data = h5py.File("/home/jsw7460/.minari/datasets/kitchen-partial-v1/data/main_data.hdf5")
buffer = GenRLDataset("/home/jsw7460/.minari/datasets/kitchen-complete-v1/data/main_data.hdf5", 777,  postprocess_fn=apply_fn)

train_dt, eval_dt = GenRLDataset.split_dataset(buffer, sizes=[8, 1])
x = train_dt.sample_subtrajectories(3, 2)


@hydra.main(version_base=None, config_path="../config/train", config_name="base")
def program(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)  # type: Dict
    env = buffer.recover_environment()

    modules_dict = {}
    for module in cfg["algo"]["modules"]:
        modules_dict[module] = instantiate(cfg[module])

    algo_cls = get_class(cfg["algo"]["cls"])
    algo = algo_cls(cfg=cfg, env=env, **modules_dict)   # type: BC

    algo.train_dataset = train_dt
    algo.eval_dataset = eval_dt

    algo.train()


if __name__ == "__main__":
    program()
