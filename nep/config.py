"""
* using ns:
    when using ns, only `-pr -m nep1` is needed. no ds config and dsname specification.
"""

from rich import print
import argparse
from omegaconf import OmegaConf
from nep.utils.base_utils import dict_merge, load_cfg
from typing import Any, Dict
import os
from pathlib import Path
import shlex


def valid_file(arg, return_path=True, root="./"):
    base_path = Path(root) / Path("configs/model")
    full_path = base_path / f"{arg}.yaml"
    if full_path.is_file() and full_path.exists():
        if return_path:
            return str(full_path)
        else:
            return arg
    else:
        raise argparse.ArgumentTypeError(f"The config file {full_path} does not exist!")


def get_config(command_string=None, root="./nep") -> Dict:
    """
    command_string: if None, parase from the real command line.
    """

    flag2yaml = {
        "pr": "configs/plugin/real-ds-plugin.yaml",
        "pB": "configs/plugin/bunny-plugin.yaml",
        "pM": "configs/plugin/bunny-plugin.yaml",
        "pH": "configs/plugin/horse-angel-plugin.yaml",
    }

    parser = argparse.ArgumentParser()
    for flag in flag2yaml.keys():
        parser.add_argument(f"-{flag}", action="store_true")

    def valid_fn(x):
        return valid_file(x, root=root)

    parser.add_argument(f"--model", "-m", type=valid_fn, required=True)
    parser.add_argument(f"--render", action="store_true")
    # parser.add_argument(f'--ckpt', type=str, help="useed for mesh extraction")

    if command_string:
        args, unknown = parser.parse_known_args(shlex.split(command_string))
    else:
        args, unknown = parser.parse_known_args()
    print("+ Cli args:", unknown)

    # ds and plugins will overwrite the model configuration
    sub_config_paths = [
        Path(root) / config_path for flag, config_path in flag2yaml.items() if getattr(args, flag)
    ]
    print(f"+ Merging sub configs: {sub_config_paths}")
    sub_configs = [load_cfg(config_path) for config_path in sub_config_paths]

    # model config will override others
    print(f"+ Merging model config: {args.model}")
    config = OmegaConf.merge(*sub_configs, load_cfg(args.model))

    # merge cli configs & resolve
    cli_cfg = OmegaConf.from_cli(unknown)
    # config.name = 'undefined'
    dict_merge(config, cli_cfg, warn_new_key=True)
    # if config.name == 'undefined':
    #     config.name = config.dsname + '-' + Path(args.model).stem

    # you must pass these args from cli:
    # if 'mesh' in config:
    #     assert config.mesh != 'undefined'
    if "bg_model" in config and "ckpt" in config.bg_model:
        assert config.bg_model.ckpt != "undefined"
    # assert config.dsname != 'undefined'

    config = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    # config['render'] = args.render
    # config['ckpt'] = args.ckpt
    return config


if __name__ == "__main__":
    print(get_config())
