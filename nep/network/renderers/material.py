from nep.config import get_config
from nerfstudio.utils.writer import EVENT_WRITERS
from nerfstudio.cameras.rays import RayBundle
import numpy as np
import torch
from typing import Any, Mapping, Optional
import torch.nn as nn
import torch.nn.functional as F
from rich import print

from nep.network.fields.mc import MCShadingNetwork
from nep.utils.base_utils import freeze_model, load_cfg, merge_config
from omegaconf import OmegaConf
from nep.network.renderers.utils import get_nep_cfg_str, ckpt_to_nep_config
from nep.network.renderers.shape import NePShapeRenderer


class NePMaterialRenderer(nn.Module):
    default_cfg = {
        "train_ray_num": 512,
        "test_ray_num": 1024,
        "rgb_loss": "charbonier",
        "reg_mat": True,
        "reg_diffuse_light": True,
        "reg_diffuse_light_lambda": 0.1,
        "fixed_camera": False,
        "valid_num": 1,
        "bounds_type": "sphere",
        "shader_cfg": {
            "diffuse_sample_num": 512,
            "specular_sample_num": 256,
            "light_exp_max": 5.0,
            "inner_light_exp_max": 5.0,
            "single_specular_light": False,
            "human_light": False,
            "geometry_type": "schlick",
            "reg_change": True,
            "change_eps": 0.05,
            "change_type": "gaussian",
            "reg_lambda1": 0.005,
            "reg_scales": [1, 1, 1],
            "reg_min_max": True,
            "random_azimuth": True,
            "is_real": False,
            "inner_light": "mlp",  # mlp, nerf, neus
            "plen_light": None,
            "nerf_sampler": "nep_bg",
            "outer_light_version": ["direction"],
            "nerf_use_linear": False,
            "outer_light_reduce": "mean",
            "outer_light_act": "exp",
            "outer_mlp_scale": 1.0,
            "outer_light_scale": 1.0,
            "mc_light_act": "exp",
            "clip_light": None,
            "less_spec": False,
            "with_stage1_mlps": False,
            "correct_schlick": False,
            "detach_roughness": False,
            "max_tan": 0.26794919243,
            "squared_roughness_fn": "identity",  # how to get the roughness ** 2 from the output of the roughness MLP.
            "roughness_transform": "lambda x: x",
            "freeze_neus": True,
            "max_vis_ray_num": 32,
            "outer_shrink": 0.999,
            "bake_loss": 0,
            "bake_every": 10,
        },
        "loss": {
            "ex": ["nerf_render", "mat_reg"],
            # "val_metric": ["mat_render"],
            # "key_metric_name": "psnr",
        },
        "bg_model": {"ckpt": None, "enabled": False, "freeze_nerf": False, "ray_samples": None},
    }

    def __init__(self, cfg, trace_fn, scene_box=None, num_train_data=None):
        # self.cfg = {**self.default_cfg, **cfg}
        # self.cfg: Any = OmegaConf.merge(self.default_cfg, cfg)
        self.cfg: Any = merge_config(self.default_cfg, cfg)
        print("NePMaterialRenderer init:")
        print(self.cfg)
        self.bg_cfg = (
            self.cfg["bg_model"]
            if self.cfg.get("bg_model") is not None and self.cfg.bg_model.get("enabled")
            else None
        )
        super().__init__()
        self.warned_normal = False
        self.trace_fn = trace_fn
        # self._init_geometry()
        # self._init_dataset(is_train)
        self._init_shader()

    def get_anneal_val(self, step):
        # just to fit the exposed ns_render api, no usage.
        return None

    def _load_stage1_model(self):
        assert isinstance(self.bg_cfg, Mapping)
        ckpt = torch.load(self.bg_cfg["ckpt"])

        # load from a ns checkpoint
        assert "pipeline" in ckpt
        print(f'==> Loading stage 1 from: {self.bg_cfg["ckpt"]}')
        config = ckpt_to_nep_config(self.bg_cfg["ckpt"])

        # remove prefix in state_dict
        state_dict = {k.replace("_model.network.", ""): v for k, v in ckpt["pipeline"].items()}
        state_dict.pop("_model.device_indicator_param")

        # load params
        # note: if nerf is trained using a kind of sampling strategy, it must use the same strategy in stage 2.
        config["nerf"]["ckpt"] = None
        model = NePShapeRenderer(config)
        model.load_state_dict(state_dict)
        if self.bg_cfg.get("ray_samples") is not None:
            model.cfg.update(self.bg_cfg["ray_samples"])

        freeze_nerf = self.bg_cfg["freeze_nerf"]
        if freeze_nerf != False:
            print(f"==> Freeze the pretrained outer_nerf: {freeze_nerf}")
            assert freeze_nerf in [True, "all", "geo"]
            if freeze_nerf in [True, "all"]:
                freeze_model(model.outer_nerf)

            elif freeze_nerf == "geo":
                freeze_model(model.outer_nerf.pts_linears)
                freeze_model(model.outer_nerf.density_linear)
                freeze_model(model.outer_nerf.feature_linear)
                if model.outer_nerf.use_refnerf:
                    freeze_model(model.outer_nerf.normal_linear)

        # load param for outer_nerf
        # outer_nerf_param = extrac_sub_state_dict(checkpoint['network_state_dict'], 'outer_nerf')
        # self.outer_nerf = NeRFNetwork(D=8, d_in=4, d_in_view=3, W=256, multires=10, multires_view=4, output_ch=4, skips=[4], use_viewdirs=True)
        # self.outer_nerf.load_state_dict(outer_nerf_param)

        print(f'==> Loading outer_nerf from step {ckpt["step"]}: {self.bg_cfg["ckpt"]}')
        return model

    def _init_shader(self):
        # load pretrained stage1 network
        self.bg_model: Optional[NePShapeRenderer] = None
        if self.bg_cfg and self.bg_cfg["enabled"]:
            if self.bg_cfg["ckpt"]:
                self.bg_model = self._load_stage1_model()

        # init shader network
        # self.cfg["shader_cfg"]["is_real"] = self.cfg["database_name"].startswith("real")
        self.shader_network = MCShadingNetwork(
            self.cfg["shader_cfg"], self.trace_fn, bg_model=self.bg_model
        )

    def _warn_ray_tracing(self, centers):
        centers = centers.reshape([-1, 3])
        distance = torch.norm(centers, dim=-1) + 1.0
        max_dist = torch.max(distance).cpu().numpy()
        if max_dist > 10.0:
            print(
                f"warning!!! the max distance from the camera is {max_dist:.4f}, which is beyond 10.0 for the ray tracer"
            )

    def shade(self, pts, view_dirs, normals, human_poses, is_train, step=None):
        ray_rgb, outputs = self.shader_network(pts, view_dirs, normals, human_poses, step, is_train)
        outputs["ray_rgb"] = ray_rgb.clamp(0.0, 1.0)
        return outputs

    # expose for ns_model
    def compute_rgb_loss(self, ray_rgb, rgb_gt, extras):
        if self.cfg["rgb_loss"] == "l1":
            rgb_loss = torch.sum(F.l1_loss(ray_rgb, rgb_gt, reduction="none"), -1)
        elif self.cfg["rgb_loss"] == "charbonier":
            epsilon = 0.001
            rgb_loss = torch.sqrt(torch.sum((rgb_gt - ray_rgb) ** 2, dim=-1) + epsilon)
        else:
            raise NotImplementedError
        return rgb_loss

    def compute_diffuse_light_regularization(self, diffuse_lights):
        diffuse_white_reg = torch.sum(
            torch.abs(diffuse_lights - torch.mean(diffuse_lights, dim=-1, keepdim=True)),
            dim=-1,
        )
        return diffuse_white_reg * self.cfg["reg_diffuse_light_lambda"]

    def ns_render(
        self,
        raybundle,
        perturb_overwrite=-1,
        cos_anneal_ratio=0.0,
        is_train=True,
        step: Optional[int] = None,
        skip_empty=False,
    ):
        human_poses = raybundle.metadata["human_poses"].view(-1, 3, 4)
        shade_outputs = self.shade(
            raybundle.metadata["inters"],
            -raybundle.directions,
            raybundle.metadata["normals"],
            human_poses,
            is_train,
            step,
        )

        # get extra losses
        if self.cfg["reg_mat"]:
            shade_outputs["loss_mat_reg"] = self.shader_network.material_regularization(
                raybundle.metadata["inters"],
                raybundle.metadata["normals"],
                shade_outputs["metallic"],
                shade_outputs["roughness"],
                shade_outputs["albedo"],
                step,
            )
        if self.cfg["reg_diffuse_light"]:
            shade_outputs["loss_diffuse_light"] = self.compute_diffuse_light_regularization(
                shade_outputs["diffuse_light"]
            )

        # log
        if self.training and step % 100 == 0:
            for k in ["metallic", "roughness"]:
                EVENT_WRITERS[0].tb_writer.add_histogram(f"Hist/{k}", shade_outputs[k], step)

        return shade_outputs
