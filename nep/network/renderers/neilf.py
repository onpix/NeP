#
from typing import Any, Mapping, Optional, Tuple, List, Callable, Dict
from nep.utils.base_utils import freeze_model, map_range_val, merge_config

# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np

from nep.network.renderers.neilf_utils.ray_sampling import fibonacci_sphere_sampling
from nep.network.renderers.neilf_utils.nn_arch import NeILFMLP, BRDFMLP, SeparatedBRDFMLP
from nep.network.renderers.neilf_utils.general import soft_clip, scale_grad, demask
from nep.network.renderers.neilf_utils.geometry import equirectangular_proj, sph2cart
from nep.network.renderers.neilf_utils.geo_volsdf import Geometry

EPS = 1e-7


class NeILFPBR(nn.Module):
    def __init__(self, config_model):
        super().__init__()
        self.S = config_model["num_train_incident_samples"]
        self.S_evel = config_model["num_eval_incident_samples"]
        separate_brdf = config_model.get("separate_brdf", True)
        # implicit brdf
        if separate_brdf:
            self.brdf_nn = SeparatedBRDFMLP(**config_model["brdf_network"])
        else:
            self.brdf_nn = BRDFMLP(**config_model["brdf_network"])
        # implicit incident light
        self.neilf_nn = NeILFMLP(**config_model["neilf_network"])

        print("Number of training incident samples: ", self.S)
        print("Number of evaluation incident samples: ", self.S_evel)

    def sample_brdfs(self, points, is_gradient=False):
        points.requires_grad_(True)

        x = self.brdf_nn(points)  # [N, 5]
        # convert from [-1,1] to [0,1]
        x = x / 2 + 0.5  # [N, 5]
        base_color, roughness, metallic = x.split((3, 1, 1), dim=-1)  # [N, 3], [N, 1], [N, 1]

        # gradients w.r.t. input position
        gradients = []
        for brdf_slice in [roughness, metallic]:
            if is_gradient:
                d_output = torch.ones_like(brdf_slice, requires_grad=False)
                gradient = torch.autograd.grad(
                    outputs=brdf_slice,
                    inputs=points,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                    allow_unused=True,
                )[0]
            else:
                gradient = torch.zeros_like(points)
            gradients.append(gradient)
        gradients = torch.stack(gradients, dim=-2)  # [N, 2, 3]

        return base_color, roughness, metallic, gradients

    def sample_incident_rays(self, normals):
        if self.training:
            incident_dirs, incident_areas = fibonacci_sphere_sampling(
                normals, self.S, random_rotate=True
            )
        else:
            incident_dirs, incident_areas = fibonacci_sphere_sampling(
                normals, self.S_evel, random_rotate=False
            )

        return incident_dirs, incident_areas  # [N, S, 3], [N, S, 1]

    def sample_incident_lights(self, points, incident_dirs):
        # point: [N, 3], incident_dirs: [N, S, 3]
        N, S, _ = incident_dirs.shape
        points = points.unsqueeze(1).repeat([1, S, 1])  # [N, S, 3]
        nn_inputs = torch.cat([points, -incident_dirs], axis=2)  # [N, S, 6]
        nn_inputs = nn_inputs.reshape([-1, 6])  # [N * S, 6]
        incident_lights = self.neilf_nn(nn_inputs)  # [N * S, 3]
        # use mono light
        if incident_lights.shape[1] == 1:
            incident_lights = incident_lights.repeat([1, 3])
        return incident_lights.reshape([N, S, 3])  # [N, S, 3]

    def plot_point_env(self, point, width):
        # incident direction of all pixels in the env map
        eval_sph, valid_mask = equirectangular_proj(width, meridian=0)  # [H, W, 2]
        eval_sph = eval_sph.to(point.device)
        valid_mask = valid_mask.to(point.device)
        eval_cart = sph2cart(eval_sph, dim=-1)  # [H, W, 3]
        eval_cart_flat = -1 * eval_cart.view([-1, 3])  # [N, 3]

        point = point.unsqueeze(0).repeat([eval_cart_flat.shape[0], 1])  # [N, 3]
        nn_inputs = torch.cat([point, eval_cart_flat], axis=1)  # [N, 6]
        env_map = self.neilf_nn(nn_inputs).view(-1, width, 3)  # [N, 3]

        env_map *= valid_mask
        return env_map

    def rendering_equation(
        self,
        output_dirs,
        normals,
        base_color,
        roughness,
        metallic,
        incident_lights,
        incident_dirs,
        incident_areas,
    ):
        # extend all inputs into shape [N, 1, 1/3] for multiple incident lights
        output_dirs = output_dirs.unsqueeze(dim=1)  # [N, 1, 3]
        normal_dirs = normals.unsqueeze(dim=1)  # [N, 1, 3]
        base_color = base_color.unsqueeze(dim=1)  # [N, 1, 3]
        roughness = roughness.unsqueeze(dim=1)  # [N, 1, 1]
        metallic = metallic.unsqueeze(dim=1)  # [N, 1, 1]

        def _dot(a, b):
            return (a * b).sum(dim=-1, keepdim=True)  # [N, 1, 1]

        def _f_diffuse(h_d_o, n_d_i, n_d_o, base_color, metallic, roughness):
            return (1 - metallic) * base_color / np.pi  # [N, 1, 1]

        def _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic):
            # used in SG, wrongly normalized
            def _d_sg(r, cos):
                r2 = torch.clamp(r**4, min=EPS)
                amp = 1 / (r2 * np.pi)
                sharp = 2 / r2
                return amp * torch.exp(sharp * (cos - 1))

            # def _d_ggx(r, cos):
            #     r2 = torch.clamp(r**4, min=EPS)
            #     return r2 / np.pi / ((r2-1) * cos**2 + 1)**2
            D = _d_sg(roughness, h_d_n)

            # Fresnel term F
            F_0 = 0.04 * (1 - metallic) + base_color * metallic  # [N, 1, 3]
            F = F_0 + (1.0 - F_0) * ((1.0 - h_d_o) ** 5)  # [N, S, 1]

            # geometry term V, we use V = G / (4 * cos * cos) here
            # def _v_schlick_ggx(r, cos):
            #     r2 = ((1 + r) ** 2) / 8
            #     return 0.5 / torch.clamp(cos * (1 - r2) + r2, min=EPS)
            def _v_schlick_ggx2(r, cos):
                r2 = r**2 / 2
                return 0.5 / torch.clamp(cos * (1 - r2) + r2, min=EPS)

            V = _v_schlick_ggx2(roughness, n_d_i) * _v_schlick_ggx2(roughness, n_d_o)

            return D * F * V  # [N, S, 1]

        # half vector and all cosines
        half_dirs = incident_dirs + output_dirs  # [N, S, 3]
        half_dirs = Func.normalize(half_dirs, dim=-1)  # [N, S, 3]
        h_d_n = _dot(half_dirs, normal_dirs).clamp(min=0)  # [N, S, 1]
        h_d_o = _dot(half_dirs, output_dirs).clamp(min=0)  # [N, S, 1]
        n_d_i = _dot(normal_dirs, incident_dirs).clamp(min=0)  # [N, S, 1]
        n_d_o = _dot(normal_dirs, output_dirs).clamp(min=0)  # [N, 1, 1]

        f_d = _f_diffuse(h_d_o, n_d_i, n_d_o, base_color, metallic, roughness)  # [N, 1, 3]
        f_s = _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic)  # [N, S, 3]

        rgb = ((f_d + f_s) * incident_lights * incident_areas * n_d_i).mean(dim=1)  # [N, 3]

        return rgb

    def forward(self, points, normals, view_dirs, geometry):
        # get brdf
        base_color, roughness, metallic, gradients = self.sample_brdfs(
            points, is_gradient=self.training
        )  # [N, 3]

        # sample incident rays for the input point
        incident_dirs, incident_areas = self.sample_incident_rays(
            normals.detach()
        )  # [N, S, 3], [N, S, 1]

        # sample incident lights for the input point
        incident_lights = self.sample_incident_lights(points, incident_dirs)  # [N, S, 3]

        # the rendering equation, first pass
        rgb = self.rendering_equation(
            view_dirs,
            normals,
            base_color,
            roughness,
            metallic,
            incident_lights,
            incident_dirs,
            incident_areas,
        )  # [N, 3]

        if self.training:
            trace_mask, miss_mask, trace_render_rgb, misc = geometry.trace_and_render(
                points.detach(), normals.detach(), incident_dirs, sample=1
            )
            trace_nn_rgb = incident_lights[trace_mask]
            miss_lights = None
            global_lights = None
        else:
            trace_nn_rgb = None
            trace_render_rgb = None
            miss_lights = None
            global_lights = None

        return (
            rgb,
            base_color,
            roughness,
            metallic,
            gradients,
            trace_nn_rgb,
            trace_render_rgb,
            miss_lights,
            global_lights,
        )


class NeILFModel(nn.Module):
    default_cfg = {
        "geometry_module": "model.geo_volsdf",
        "use_ldr_image": True,
        "num_train_incident_samples": 128,
        "num_eval_incident_samples": 256,
        "separate_brdf": False,
        "brdf_network": {
            "in_dims": 3,
            "out_dims": 5,
            "dims": [512, 512, 512, 512, 512, 512, 512, 512],
            "skip_connection": [4],
            "weight_norm": False,
            "embed_config": [{"otype": "Identity"}, {"otype": "Frequency", "n_frequencies": 6}],
            "use_siren": True,
        },
        "neilf_network": {
            "in_dims": 6,
            "out_dims": 1,
            "dims": [128, 128, 128, 128, 128, 128, 128, 128],
            "dir_insert": [0, 4],
            "pos_insert": [4],
            "embed_config_view": [
                {"otype": "Identity"},
                {"otype": "Frequency", "n_frequencies": 6},
            ],
            "embed_config": [{"otype": "Identity"}],
            "use_siren": True,
            "weight_norm": False,
            "init_output": -1.0,
            "fused": False,
        },
        "geometry": {
            "scene_bounding_sphere": 3.0,
            "cutoff_bounding_sphere": 2.0,
            "object_bounding_sphere": 1.0,
            "implicit_network": {
                "dims": [256, 256, 256, 256, 256, 256, 256, 256],
                "aux_dims": [["pred_grad", 3]],
                "geometric_init": True,
                "bias": 0.6,
                "skip_in": [4],
                "weight_norm": True,
                "hess_type": "analytic",
                "sphere_scale": 20.0,
                "embed_config": [{"otype": "Identity"}, {"otype": "Frequency", "n_frequencies": 6}],
            },
            "rendering_network": {
                "mode": "idr",
                "d_out": 3,
                "dims": [256, 256, 256, 256],
                "weight_norm": True,
                "embed_config_view": [
                    {"otype": "Identity"},
                    {"otype": "Frequency", "n_frequencies": 4},
                ],
                "fused": False,
                "last_act": "Sigmoid",
            },
            "density": {"params_init": {"beta": 0.1}, "beta_min": 0.0001},
            "ray_sampler": {
                "near": 0.0,
                "N_samples": 64,
                "N_samples_eval": 128,
                "N_samples_extra": 32,
                "eps": 0.1,
                "beta_iters": 10,
                "max_total_iters": 5,
                "N_samples_inverse_sphere": 32,
                "add_tiny": 1.0e-6,
            },
            "background": {
                "otype": "nerfpp",
                "implicit_network": {
                    "dims": [128, 128, 128, 128, 128, 128, 128, 128],
                    "aux_dims": [["pred_grad", 3]],
                    "skip_in": [4],
                    # "geometric_init": True,
                    # "bias": 0.6,
                    "weight_norm": True,
                    # "hess_type": "analytic",
                    # "sphere_scale": 20.0,
                    "embed_config": [
                        {"otype": "Identity"},
                        {"otype": "Frequency", "n_frequencies": 6},
                    ],
                },
                # "implicit_network": {
                #     "dims": [64, 64],
                #     "weight_norm": False,
                #     "embed_config": [
                #         {"otype": "Identity"},
                #         {
                #             "otype": "HashGrid",
                #             "n_levels": 16,
                #             "n_features_per_level": 2,
                #             "log2_hashmap_size": 19,
                #             "base_resolution": 16,
                #             "max_resolution": 4096,
                #             "interpolation": "Linear",
                #             "input_range": [-3, 3],
                #             "init_range": 1e-4,
                #         },
                #     ],
                # },
                "rendering_network": {
                    "mode": "nerf",
                    "d_out": 3,
                    "dims": [64, 64],
                    "weight_norm": False,
                    "embed_config_view": [
                        {"otype": "Identity"},
                        {"otype": "Frequency", "n_frequencies": 6}
                        # {"otype": "SphericalHarmonics", "n_frequencies": 4, "input_range": [-1, 1]},
                    ],
                    "fused": False,
                    "last_act": "Sigmoid",
                    "sigmoid_output_scale": 3.0,
                },
            },
        },
        "phase": "joint",
        "loss": {"ex": []},
        # merge train to model config
        "train": {
            "num_pixel_samples": 3072,
            "num_reg_samples": 8192,
            "training_iterations": 80000,
            "mat2geo_grad_scale": 0.1,
            "lr": 0.0003,
            "lr_scaler": 0.01,
            "lr_warmup_start_factor": 0.1,
            "lr_warmup_iters": 0,
            "lr_decay": 0.1,
            "lr_decay_iters": [40000],
            "geo_loss": {
                "rgb_loss": "torch.nn.L1Loss",
                "eikonal_weight": 0.1,
                "hessian_weight": 0.0001,
                "minsurf_weight": 0.0001,
                "orientation_weight": 0.0,
                "pred_normal_weight": 0.0,
                "feature_weight": 0.0,
                "pcd_weight": 0.0,
            },
            "rgb_loss": "torch.nn.L1Loss",
            "lambertian_weighting": 0.0000,
            "smoothness_weighting": 0.0000,
            "trace_weighting": 0.1,
            "var_weighting": 0.1,
            "remove_black": True,
            "phase": {
                "geo": {
                    "num_pixel_samples": 4096,
                    "training_iterations": 8000,
                    "mat2geo_grad_scale": 1.0,
                    "lr": 0.001,
                    "lr_decay_iters": [4000],
                    "geo_loss": {"hessian_weight": 0.001, "minsurf_weight": 0.001},
                },
                "mat": {
                    "training_iterations": 8000,
                    "mat2geo_grad_scale": 0.0,
                    "lr": 0.001,
                    "lr_decay_iters": [],
                },
            },
        },
    }

    def __init__(self, config_model, trace_fn, scene_box, num_train_data):
        super().__init__()

        if "shader_cfg" in config_model:
            config_model.pop("shader_cfg")

        self.cfg: Any = merge_config(self.default_cfg, config_model)
        print("NePShapeRenderer init:")
        print(self.cfg)
        self.phase = self.cfg.phase
        self.set_mat2geo_grad_scale(self.cfg["train"]["mat2geo_grad_scale"])

        kwargs = {}
        if self.cfg["geometry_module"] == "model.geo_fixmesh":
            from pyrt import PyRT

            ray_tracer = PyRT(*self.dataset.tracer_mesh, 0)
            kwargs["ray_tracer"] = ray_tracer

        if self.cfg["geometry_module"] == "model.geo_volsdf":
            kwargs["num_reg_samples"] = self.cfg["train"]["num_reg_samples"]
            kwargs["calc_hess"] = self.cfg["train"]["geo_loss"]["hessian_weight"] > 0

        self.geometry = Geometry(self.cfg["geometry"], self.phase, **kwargs)
        self.phase = self.cfg.phase
        self.to_ldr = self.cfg["use_ldr_image"]

        # neilf rendering
        self.neilf_pbr = NeILFPBR(self.cfg)

        # learnable gamma parameter to map HDR to LDR
        if self.to_ldr:
            # self.gamma = nn.Parameter(torch.ones(1).float().cuda())
            self.gamma = torch.tensor(0.4545).float().cuda()

        self.rgb_var = self.cfg["train"]["var_weighting"] > 0

    def set_mat2geo_grad_scale(self, scale):
        self.mat2geo_grad_scale = scale

    def plot_point_env(self, point, width):
        return self.neilf_pbr.plot_point_env(point, width), self.geometry.plot_point_apr(
            point, width
        )

    # todo: impl __init__, compute_rgb_loss, ns_render

    def ns_render(self, ray_bundle, perturb_overrite, anneal, is_train=True, step=None):
        # parse input
        (
            points,
            normals,
            view_dirs,
            geo_rgb,
            render_masks,
            total_samples,
            geo_aux_output,
        ) = self.geometry(ray_bundle)

        # modify the gradient for geometry
        points_o = points
        normals_o = normals
        if self.training:
            if isinstance(self.mat2geo_grad_scale, (tuple, list)):
                points_grad_scale = self.mat2geo_grad_scale[0]
                normals_grad_scale = self.mat2geo_grad_scale[1]
            else:
                points_grad_scale = normals_grad_scale = self.mat2geo_grad_scale
            points = scale_grad(points, points_grad_scale)
            normals = scale_grad(normals, normals_grad_scale)

        # neilf rendering
        if self.phase in ["mat", "joint"]:
            (
                rgb,
                base_color,
                roughness,
                metallic,
                gradients,
                trace_nn_rgb,
                trace_render_rgb,
                miss_lights,
                global_lights,
            ) = self.neilf_pbr(
                points[render_masks], normals[render_masks], view_dirs[render_masks], self.geometry
            )

        if self.phase == "mat" and self.rgb_var:
            rgb_var = self.geometry.radiance_variance(points[render_masks], normals[render_masks])
            rgb_var = demask(rgb_var, render_masks)

        # convert to ldr
        if self.to_ldr:
            if self.phase in ["mat", "joint"]:
                rgb = torch.clamp(rgb, EPS, 1)
                rgb = rgb**self.gamma
            geo_rgb = torch.clamp(geo_rgb, EPS, 1)
            geo_rgb = geo_rgb**self.gamma

        # demask
        if self.phase in ["mat", "joint"]:
            masked_outputs = [rgb, base_color, roughness, metallic, gradients]
            outputs = [demask(arr, render_masks) for arr in masked_outputs]
            rgb_values, base_color, roughness, metallic = outputs[:-1]

        output = {
            "render_masks": render_masks,
        }

        if self.phase in ["geo", "joint"]:
            output.update(
                {
                    "points": points_o,
                    "normals": normals_o,
                    "geo_rgb": geo_rgb,
                }
            )

        if self.phase in ["mat", "joint"]:
            output.update(
                {
                    "ray_rgb": rgb_values,
                    "base_color": base_color,
                    "roughness": roughness,
                    "metallic": metallic,
                }
            )

        if self.phase == "mat" and self.rgb_var:
            output.update(
                {
                    "rgb_var": rgb_var,
                }
            )

        if self.training:
            if self.phase in ["geo", "joint"]:
                output["geo_aux_output"] = geo_aux_output
            if self.phase in ["mat", "joint"]:
                output["brdf_grads"] = outputs[-1]
                output["trace_nn_rgb"] = trace_nn_rgb
                output["trace_render_rgb"] = trace_render_rgb
                output["miss_lights"] = miss_lights
                output["global_lights"] = global_lights
            if self.to_ldr:
                output["gamma"] = self.gamma

        output["metrics"] = {}
        return output
