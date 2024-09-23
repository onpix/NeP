import argparse
from nerfstudio.field_components.activations import trunc_exp
from nep.network.renderers.utils import ckpt_to_nep_config, ckpt2nsconfig
from nerfstudio.data.scene_box import SceneBox
import math
from pathlib import Path
import numpy as np
import torch
import trimesh
import sys
import yaml
import omegaconf
from nerfstudio.plugins.types import TrainerConfig
from nep.network.fields.utils import extract_geometry
from nep.network.renderers import name2renderer
from nep.network.renderers.utils import get_nep_cfg_str
from config import get_config
from nep.network.renderers.utils import ckpt_to_nep_config
from nep.ns.nep_dataparser import ds_meta_info, NePColmapDataParser
import argparse
from nep.network.renderers.shape import NePShapeRenderer


def _compute_rotation(up, forward):
    y = np.cross(up, forward)
    x = np.cross(y, up)

    up = up / np.linalg.norm(up)
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    R = np.stack([x, y, up], 0)
    return R


def nerfacto_density_fn(self, positions):
    """Computes and returns the densities."""
    if self.spatial_distortion is not None:
        # positions = ray_samples.frustums.get_positions()
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0
    else:
        raise NotImplementedError
        positions = SceneBox.get_normalized_positions(
            ray_samples.frustums.get_positions(), self.aabb
        )
    # Make sure the tcnn gets inputs between 0 and 1.
    selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
    positions = positions * selector[..., None]
    self._sample_locations = positions
    if not self._sample_locations.requires_grad:
        self._sample_locations.requires_grad = True
    positions_flat = positions.view(-1, 3)
    h = self.mlp_base(positions_flat).view(*positions.shape[:-1], -1)
    density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
    self._density_before_activation = density_before_activation
    density = trunc_exp(density_before_activation.to(positions))
    density = density * selector[..., None]
    return density

    # delta_density = 1e-3 * density
    # alpha = 1 - torch.exp(-delta_density)
    # return alpha


def colmap2nerf(vec):
    vec = vec[[1, 0, 2]]
    vec[-1] *= -1
    return vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path")
    # parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--nf", action="store_true")
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--th", type=float, default=5)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--offset", type=float, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output and Path(args.output).exists():
        print(f"{args.output} already exists. Skipping...")

    ckpt_path = args.ckpt_path
    resolution = 512

    ckpt = torch.load(ckpt_path)
    step = ckpt["step"]
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # for nep
    if not args.nf:
        cfg = ckpt_to_nep_config(ckpt_path)
        network = NePShapeRenderer(cfg, ignore_check=True)
        ckpt = {k.replace("_model.network.", ""): v for k, v in ckpt["pipeline"].items()}
        ckpt.pop("_model.device_indicator_param")
        network.load_state_dict(ckpt)
        network.eval().cuda()
        print(f"successfully load step {step}.")
        func = lambda x: network.sdf_network.sdf(x)
    else:
        from nep.ns.models.nf import MyNerfactoModel, MyNerfactoModelConfig

        scenebox = SceneBox(torch.Tensor([[-1, -1, -1], [1, 1, 1]]))
        param = ckpt
        nerfacto_config = ckpt2nsconfig(ckpt_path)["pipeline"]["model"]
        nerf_param = {k.replace("_model.", ""): v for k, v in param["pipeline"].items()}
        nerfacto_config = MyNerfactoModelConfig(**nerfacto_config)
        cam_embed_dim = nerf_param["field.embedding_appearance.embedding.weight"].shape[0]
        network = MyNerfactoModel(nerfacto_config, scenebox, cam_embed_dim)
        network.load_state_dict(nerf_param, strict=False)
        network.eval().cuda()
        threshold = args.th
        func = lambda x: -(
            nerfacto_density_fn(network.field, (x - args.offset) / args.scale) - threshold
        )
        # func = lambda x: nerfacto_density_fn(network.field, x)

    bbox_min = -torch.ones(3)
    bbox_max = torch.ones(3)
    suffix = "noclip-"
    with torch.no_grad():
        vertices, triangles = extract_geometry(bbox_min, bbox_max, resolution, 0, func)

    # output geometry
    mesh = trimesh.Trimesh(vertices, triangles)
    if args.output is None:
        outpath = Path(ckpt_path).parent / f"{suffix}{step}.ply"
    else:
        outpath = args.output
    mesh.export(str(outpath))
    print("Exported at:", outpath)


if __name__ == "__main__":
    main()
