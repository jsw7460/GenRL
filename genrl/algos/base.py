import pickle
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

import numpy as np

from genrl.rl.buffers.type_aliases import GenRLBufferSample
from genrl.utils.common.type_aliases import GymEnv
from genrl.utils.superclasses.loggable import Loggable


class BaseTrainer(Loggable):

    def __init__(self, cfg: Dict, env: GymEnv):
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])

        super(BaseTrainer, self).__init__(cfg=cfg)

        self.env = env

        # ==========
        #  " Time "
        # ==========
        self.today = None
        self.start = None

        self.cfg = cfg

        self.n_update = 0
        self.batch_size = cfg["batch_size"]
        self.subseq_len = cfg["subseq_len"]

        self.max_iter = cfg["max_iter"]
        self.log_interval = cfg["log_interval"]
        self.save_interval = cfg["save_interval"]
        self.eval_interval = cfg["eval_interval"]
        self.required_total_update = None

        self.prepare_run()

        self.metadata = {
            "info/suffix": self.cfg["save_suffix"],
            "date": self.today.strftime('%Y-%m-%d %H:%M:%S')
        }

    def dump_logs(self, step: int):
        now = datetime.now()
        elapsed = max((now - self.start).seconds, 1)
        fps = step / elapsed
        remain = int((self.required_total_update - step) / fps)
        eta = now + timedelta(seconds=remain)
        self.record({
            "time/fps": fps,
            "time/elapsed": str(timedelta(seconds=elapsed)),
            "time/remain": str(timedelta(seconds=remain)),
            "time/eta": eta.strftime("%m.%d / %H:%M:%S"),
            **self.metadata
        })
        super(BaseTrainer, self).dump_logs(step=step)

    def _prepare_run(self):
        pass

    def _sample_train_batch(self) -> GenRLBufferSample:
        raise NotImplementedError()

    def _update_model(self, data: GenRLBufferSample) -> Dict:
        raise NotImplementedError()

    def _evaluate_model(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError()

    def prepare_run(self):
        self._prepare_run()
        prefix = self.cfg["save_prefix"]
        suffix = self.cfg["save_suffix"]
        self.today = datetime.today()
        today_str = self.today.strftime('%Y-%m-%d')
        date_prefix = Path(prefix) / Path(today_str)

        cfg_prefix = (Path(date_prefix) / Path("cfg"))
        cfg_prefix.mkdir(parents=True, exist_ok=True)

        self.cfg["save_paths"] = dict()

        for module_key in self.cfg["modules"]:
            module_prefix = (Path(date_prefix) / Path(f"{module_key}"))
            module_prefix.mkdir(parents=True, exist_ok=True)

            module_fullpath = module_prefix / Path(suffix)
            self.cfg["save_paths"][module_key] = str(module_fullpath)

        self.cfg.update({"date": today_str, "wandb_url": self.wandb_url})

        # Dump configure file
        with open(str(cfg_prefix / Path(f"cfg_{suffix}")), "wb") as f:
            pickle.dump({**self.cfg, "env_recover": self.env}, f)

        self.start = datetime.now()

    def train(self):
        while self.n_update < self.required_total_update:
            batch_data = self._sample_train_batch()
            info = self.update_model(batch_data)
            self.record_from_dicts(info, mode="train")
            self.n_update += 1

            if (self.n_update % self.log_interval) == 0:
                self.dump_logs(step=self.n_update)

            if (self.n_update % self.save_interval) == 0:
                self.save()

            if (self.n_update % self.eval_interval) == 0:
                self.evaluate()

    def update_model(self, data: GenRLBufferSample) -> Dict:
        return self._update_model(data)

    def save(self):
        for key, save_path in self.cfg["save_paths"].items():
            cur_step = str(self.n_update)
            getattr(self, key).save(f"{save_path}_{cur_step}")

    def evaluate(self, *args, **kwargs) -> None:
        eval_dict = self._evaluate_model(*args, **kwargs)
        self.record_from_dicts(eval_dict, mode="eval")
