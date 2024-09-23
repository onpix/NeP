import cv2
import re
import yaml
import numpy as np
import torch
from pathlib import Path
from rich import print

from nep.dataset.database import parse_database_name, get_database_split, BaseDatabase
from nep.utils.base_utils import (
    color_map_forward,
    downsample_gaussian_blur,
)


def ckpt2nsconfig(ckpt_path):
    # Read the YAML file content
    file_path = Path(ckpt_path).parent / "config.yml"
    with open(file_path, "r") as file:
        content = file.readlines()

    # Remove unwanted patterns from each line
    cleaned_content = []
    for line in content:
        cleaned_line = re.sub(r"!!.*|&id.*|_target.*|data: \*id003", "", line)
        cleaned_content.append(cleaned_line)
        # print("before:", line, "after:", cleaned_line)

    # Join the cleaned lines and parse the YAML content
    cleaned_yaml_str = "".join(cleaned_content)
    return yaml.safe_load(cleaned_yaml_str)


def get_nep_cfg_str(ns_ckpt_path):
    ns_config_path = Path(ns_ckpt_path).parent / "config.yml"

    with open(ns_config_path, "r") as f:
        for line in f:
            if "nep_cfg:" in line:
                # Split by the colon and strip whitespace to get the value
                res = line.split("nep_cfg:")[1].strip().replace("'", "").replace('"', "")
                print("Found nep-cfg:", res)
                return res
    return None


def ckpt_to_nep_config(ns_ckpt_path):
    return yaml.load(
        open(Path(ns_ckpt_path).parent / "nep-config.yaml"),
        yaml.FullLoader,
    )


def build_imgs_info(database: BaseDatabase, img_ids):
    images = [database.get_image(img_id) for img_id in img_ids]
    poses = [database.get_pose(img_id) for img_id in img_ids]
    Ks = [database.get_K(img_id) for img_id in img_ids]

    images = np.stack(images, 0)
    images = color_map_forward(images).astype(np.float32)
    Ks = np.stack(Ks, 0).astype(np.float32)
    poses = np.stack(poses, 0).astype(np.float32)
    return {"imgs": images, "Ks": Ks, "poses": poses}


def imgs_info_to_torch(imgs_info, device="cpu"):
    for k, v in imgs_info.items():
        v = torch.from_numpy(v)
        if k.startswith("imgs"):
            v = v.permute(0, 3, 1, 2)
        imgs_info[k] = v.to(device)
    return imgs_info


def imgs_info_slice(imgs_info, idxs):
    new_imgs_info = {}
    for k, v in imgs_info.items():
        new_imgs_info[k] = v[idxs]
    return new_imgs_info


def imgs_info_to_cuda(imgs_info):
    for k, v in imgs_info.items():
        imgs_info[k] = v.cuda()
    return imgs_info


def imgs_info_downsample(imgs_info, ratio):
    b, _, h, w = imgs_info["imgs"].shape
    dh, dw = int(ratio * h), int(ratio * w)
    imgs_info_copy = {k: v for k, v in imgs_info.items()}
    imgs_info_copy["imgs"], imgs_info_copy["Ks"] = [], []
    for bi in range(b):
        img = imgs_info["imgs"][bi].cpu().numpy().transpose([1, 2, 0])
        img = downsample_gaussian_blur(img, ratio)
        img = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_LINEAR)
        imgs_info_copy["imgs"].append(torch.from_numpy(img).permute(2, 0, 1))
        K = torch.from_numpy(np.diag([dw / w, dh / h, 1]).astype(np.float32)) @ imgs_info["Ks"][bi]
        imgs_info_copy["Ks"].append(K)

    imgs_info_copy["imgs"] = torch.stack(imgs_info_copy["imgs"], 0)
    imgs_info_copy["Ks"] = torch.stack(imgs_info_copy["Ks"], 0)
    return imgs_info_copy
