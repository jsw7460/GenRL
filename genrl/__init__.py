import logging
import os
import warnings

os.environ["JAX_IGNORE_TPU_DEVICES"] = "1"
warnings.simplefilter("ignore", UserWarning)
logging.getLogger('jax._src.lib.xla_bridge').addFilter(lambda _: False)
