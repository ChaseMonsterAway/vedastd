<<<<<<< HEAD
from .config import ConfigDict, Config
from .common import build_from_cfg, get_root_logger, set_random_seed
from .registry import Registry
from .checkpoint import load_state_dict, load_checkpoint, save_checkpoint
=======
from .config import Config
from .registry import Registry, build_from_cfg
from .checkpoint import load_checkpoint, save_checkpoint, weights_to_cpu
>>>>>>> origin/yuxin
