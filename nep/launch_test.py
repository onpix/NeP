from train.trainer import Trainer
from config import get_config
cmd = "-dr -pr -m nep1_mip dsname=vase"
Trainer(get_config(cmd)).run()

