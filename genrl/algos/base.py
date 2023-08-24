import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

from genrl.utils.common.type_aliases import GymEnv
from genrl.utils.superclasses.loggable import Loggable


class BaseTrainer(Loggable):

    def __init__(self, cfg: Dict, env: GymEnv):
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

        self.required_total_update = None

        self.prepare_run()

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
            "time/eta": eta.strftime("%m.%d / %H:%M:%S")
        })
        super(BaseTrainer, self).dump_logs(step=step)

    def _prepare_run(self):
        pass

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

        self.cfg.update({
            "date": today_str,
            "wandb_url": self.wandb_url
        })

        # Dump configure file
        with open(str(cfg_prefix / Path(f"cfg_{suffix}")), "wb") as f:
            pickle.dump(self.cfg, f)

        self.start = datetime.now()
