from nerfstudio.cameras.rays import RayBundle
import yaml
from nerfstudio.field_components.field_heads import FieldHeadNames
from nep.ns.models.nf import NerfactoModel, MyNerfactoModelConfig
import math
import nerfacc
from nerfacc import render_weight_from_alpha, accumulate_along_rays
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
import itertools
from nerfstudio.model_components.ray_samplers import (
    PDFSampler,
    UniformSampler,
    ProposalNetworkSampler,
    NeuSSampler,
)
import open3d
import numpy as np
import torch
from typing import Any, Mapping, Optional, Tuple, List, Callable, Dict
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from torch import Tensor
from nerfstudio.models.nerfacto import SceneContraction
from nerfstudio.data.scene_box import SceneBox
from nep.network.fields.utils import get_intersection, extract_geometry, sample_pdf
from nep.network.fields.app import AppShadingNetwork
from nep.network.fields.nerf import NeRFNetwork
from nep.network.fields.neus import SDFNetwork, SingleVarianceNetwork
from nerfacc.estimators.prop_net import get_proposal_requires_grad_fn
from nep.utils.raw_utils import linear_to_srgb
from nep.utils.base_utils import freeze_model, map_range_val, merge_config
from nep.network.renderers.utils import ckpt_to_nep_config, ckpt2nsconfig
from nep.network.loss import pred_normal_loss, orientation_loss
from omegaconf import OmegaConf


class MyProposalNetworkSampler(ProposalNetworkSampler):
    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        density_fns: Optional[List[Callable]] = None,
        use_gaussian: bool = True,
    ) -> Tuple[RaySamples, List, List]:
        assert ray_bundle is not None
        assert density_fns is not None

        weights_list = []
        ray_samples_list = []

        n = self.num_proposal_network_iterations
        weights = None
        ray_samples = None
        updated = self._steps_since_update > self.update_sched(self._step) or self._step < 10
        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = (
                self.num_proposal_samples_per_ray[i_level]
                if is_prop
                else self.num_nerf_samples_per_ray
            )
            if i_level == 0:
                # Uniform sampling because we need to start with some samples
                ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)
            else:
                # PDF sampling based on the last samples and their weights
                # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                assert weights is not None
                annealed_weights = torch.pow(weights, self._anneal)
                ray_samples = self.pdf_sampler(
                    ray_bundle, ray_samples, annealed_weights, num_samples=num_samples
                )
            if is_prop:
                # note: use gaussians, not xyz
                if use_gaussian:
                    gaussians = ray_samples.frustums.get_gaussian_blob()
                    gaussians.mean = gaussians.mean.detach()
                    gaussians.cov = gaussians.cov.detach()
                    density_fn_input = gaussians
                else:
                    density_fn_input = ray_samples.frustums.get_positions()

                if updated:
                    # always update on the first step or the inf check in grad scaling crashes
                    density = density_fns[i_level](density_fn_input)
                else:
                    with torch.no_grad():
                        density = density_fns[i_level](density_fn_input)
                weights = ray_samples.get_weights(density)
                weights_list.append(weights)  # (num_rays, num_samples)
                ray_samples_list.append(ray_samples)
        if updated:
            self._steps_since_update = 0

        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list


class NePShapeRenderer(nn.Module):
    default_cfg = {
        # standard deviation for opacity density
        "std_net": "default",
        "std_act": "exp",
        "inv_s_init": 0.3,
        "freeze_inv_s_step": None,
        # geometry network
        "bounds_type": "sphere",
        "sdf_activation": "none",
        "sdf_bias": 0.5,
        "sdf_n_layers": 8,
        "sdf_freq": 6,
        "sdf_d_out": 257,
        "geometry_init": True,
        # shader network
        "shader_cfg": {
            "is_real": True,
            "outer_light_version": None,
            "use_bg_model": False,
            "bg_sampler": "uniform",
            "human_light": False,
            "sphere_direction": False,
            "light_pos_freq": 8,
            "inner_init": -0.95,
            "roughness_init": 0.0,
            "metallic_init": 0.0,
            "light_exp_max": 0.0,
            "use_refneus": False,
            "lr_decomp": False,
            "refneus_with_app": False,
            "refneus_single_mlp": False,
            "refneus_single_dir_mlp": False,
            "refneus_with_refdir": True,
            "deep_inner_light_mlp": False,
            "light_act": "exp",
            "release": False,
            "d_enc": "ide",
            "outer_shrink": None,
        },
        # sampling strategy
        "n_samples": 64,
        "n_bg_samples": 16,
        "n_bg_samples_importance": 32,
        "inf_far": 1000.0,
        "n_importance": 64,
        "up_sample_steps": 4,  # 1 for simple coarse-to-fine sampling
        "perturb": 1.0,
        "anneal_end": 50000,
        "train_ray_num": 512,
        "test_ray_num": 1024,
        "clip_sample_variance": True,
        # dataset
        "database_name": "nerf_synthetic/lego/black_800",
        # validation
        "test_downsample_ratio": True,
        "downsample_ratio": 0.25,
        "val_geometry": False,
        # losses
        "rgb_loss": "charbonier",
        "apply_occ_loss": True,
        "occ_loss_step": 20000,
        "occ_loss_max_pn": 2048,
        "occ_sdf_thresh": 0.01,
        "bake_neus_to_nerf": False,
        "bake_every": 100,
        "bake_loss_type": "l2",
        "unbiased_sigma": False,
        "coarse_rgb_weight": 0.1,
        "fixed_camera": False,
        "valid_num": 1,
        "grid": {
            "enabled": False,
        },
        "nerf": {
            "type": "nep",  # nep for nep outer nerf, nf for nerfacto
            "ckpt": None,
            "freeze_sigma": True,  # if ckpt provided, freeze the nerf
            "freeze_rgb": True,
            "freeze_grid": True,
            "near": 0.1,
            "far": 1000,
            "use_refnerf": False,  # input reflection dir for nerf
            "contract_bg": False,
            "use_viewdirs": True,  # if false, color is fully dependent on xyz
            "sampler": "nep_bg",
            "rgb_exp": True,
            "din": 4,  # if not mip, the value of input dim
            "mip": False,
            "norm_360_input": True,
            "use_sh": False,
        },
        "ref_score_detach": False,
        "ref_score_type": "l2",
        "score_weight_type": "div",
        "score_weight_max": 1.0,
        "eikonal_weight": 0.1,
        "loss": {
            "normal_smooth": 0,
            "inner": {
                "weight": 0,
                "weight_detach": True,
                "gt_detach": False,
            },
            "curvature": {
                "weight": 0,
                "reduce_start": 50000,
                "reduce_step": 2000,
            },
            "orientation": 0,
            "pred_normal": 3e-4,
            "convex": {
                "weight": 0,
            },
            "ex": ["nerf_render", "eikonal", "std", "init_sdf_reg"],
        },
    }

    def __init__(self, cfg, trace_fn=None, **kargs):
        super().__init__()
        assert trace_fn is None
        # self.cfg = {**self.default_cfg, **cfg}
        # self.cfg: Any = OmegaConf.merge(self.default_cfg, cfg)
        # ignore_check=kargs["ignore_check"] if "ignore_check" in kargs else False,
        self.cfg: Any = merge_config(self.default_cfg, cfg)
        print("NePShapeRenderer init:")
        print(self.cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mip = self.cfg.nerf.mip
        # self.use_mipneus = self.cfg["use_mipneus"]

        # neus
        self.sdf_network = SDFNetwork(
            d_out=self.cfg["sdf_d_out"],
            d_in=3,
            d_hidden=256,
            n_layers=self.cfg["sdf_n_layers"],
            skip_in=[self.cfg["sdf_n_layers"] // 2],
            multires=self.cfg["sdf_freq"],
            bias=self.cfg["sdf_bias"],
            scale=1.0,
            geometric_init=self.cfg["geometry_init"],
            weight_norm=True,
            sdf_activation=self.cfg["sdf_activation"],
            # use_mipneus=self.use_mipneus,
        )
        self.deviation_network = SingleVarianceNetwork(
            init_val=self.cfg["inv_s_init"], activation=self.cfg["std_act"]
        )

        # nerf
        n_nerf_samples = self.cfg["n_bg_samples"]
        self.bg_sampler_uniform = UniformSampler(num_samples=n_nerf_samples)
        self.bg_sampler_pdf = PDFSampler(
            num_samples=self.cfg["n_bg_samples_importance"], include_original=False
        )
        self.neus_sampler = NeuSSampler(
            num_samples=self.cfg["n_samples"],
            num_samples_importance=self.cfg["n_importance"],
            num_samples_outside=None,
            num_upsample_steps=self.cfg["up_sample_steps"],
            # base_variance=,
        )

        prop_warmup = 5000
        prop_update_every = 5

        self.proposal_sampler = MyProposalNetworkSampler(
            num_nerf_samples_per_ray=n_nerf_samples,
            num_proposal_samples_per_ray=[n_nerf_samples * 2] * 2,
            num_proposal_network_iterations=2,
            single_jitter=True,
            update_sched=lambda step: np.clip(
                np.interp(step, [0, prop_warmup], [0, prop_update_every]), 1, prop_update_every
            ),
            # initial_sampler=self.bg_sampler_uniform,
        )

        create_nerf = lambda D, W, density_only: NeRFNetwork(
            D=D,
            d_in=3 if self.use_mip else self.cfg.nerf.din,
            d_in_view=3,
            W=W,
            multires=10,
            multires_view=4,
            skips=[4],
            use_viewdirs=self.cfg.nerf.use_viewdirs,
            density_only=density_only,
            ns=self.use_mip,
            spatial_distortion=SceneContraction(order=float("inf"))
            if self.cfg.nerf.contract_bg
            else None,
            use_refnerf=self.cfg.nerf.use_refnerf,
            norm_360_input=self.cfg.nerf.norm_360_input,
            use_sh=self.cfg.nerf.use_sh,
        )
        self.outer_nerf: Union[NerfactoModel, NeRFNetwork]
        if self.cfg.nerf.type == "nep":
            self.outer_nerf = create_nerf(8, 256, False)
        elif self.cfg.nerf.type == "nf":
            nerfacto_config = yaml.safe_load(open("./nep/configs/model/nf.yaml"))
            nerfacto_config = MyNerfactoModelConfig(**nerfacto_config)
            self.outer_nerf = NerfactoModel(
                nerfacto_config, kargs["scene_box"], kargs["num_train_data"]
            )
        else:
            raise NotImplementedError

        # nerfacc
        if self.cfg["shader_cfg"]["use_bg_model"] or self.cfg.nerf.sampler == "grid":
            self.estimator_nerf = nerfacc.OccGridEstimator(roi_aabb=[-1, -1, -1, 1, 1, 1], levels=4)
            # self.estimator_neus = nerfacc.OccGridEstimator(roi_aabb=[-1, -1, -1, 1, 1, 1], levels=1)
            self.estimator_neus = None
        else:
            self.estimator_nerf = self.estimator_neus = None

        if self.cfg.nerf.sampler == "prop":
            # self.proposal_requires_grad_fn = get_proposal_requires_grad_fn()
            self.proposal_networks = nn.ModuleList(
                [
                    create_nerf(2, 128, False),
                    create_nerf(2, 256, False),
                ]
            )
            self.density_fns = [
                lambda input: (
                    fn(input.mean, None, input.cov, density_only=True)["density"]
                    if self.use_mip
                    else fn(input, None, None, density_only=True)["density"]
                )
                for fn in self.proposal_networks
            ]

        # background nerf is a nerf++ model (this is outside the unit bounding sphere, so we call it outer nerf)
        # the input dim of nerf is 4, because it appends the norm of the positions

        if self.cfg.nerf.ckpt is not None and self.cfg.nerf.type == "nep":
            # this ckpt should be NeRFRenderer class, trained using nerfstudio
            print(f"==> Loading background nerf from {self.cfg.nerf.ckpt}")
            param = torch.load(self.cfg.nerf.ckpt)
            nerf_param = {
                k.replace("_model.network.outer_nerf.", ""): v
                for k, v in param["pipeline"].items()
                if "outer_nerf" in k
            }
            nerf_config = ckpt_to_nep_config(self.cfg.nerf.ckpt)

            # make sure the arguments about nerf is the same
            for k, v in nerf_config["nerf"].items():
                if k != "ckpt" and not k.startswith("freeze"):
                    assert v == self.cfg.nerf[k], f"{k}: {v} != {self.cfg.nerf[k]}"
            self.outer_nerf.load_state_dict(nerf_param)

            # freeze
            if self.cfg.nerf.freeze_sigma and self.cfg.nerf.freeze_rgb:
                print("Freeze outer nerf.")
                assert self.cfg.nerf.freeze_grid
                freeze_model(self.outer_nerf)

            elif self.cfg.nerf.freeze_sigma:
                print("Freeze outer density layers only for outer nerf.")
                freeze_model(self.outer_nerf.pts_linears)
                freeze_model(self.outer_nerf.density_linear)
                freeze_model(self.outer_nerf.feature_linear)
                if self.outer_nerf.use_refnerf:
                    freeze_model(self.outer_nerf.normal_linear)

            elif self.cfg.nerf.freeze_rgb:
                print("Freeze outer rgb layers only for outer nerf.")
                freeze_model(self.outer_nerf.views_linears)
                freeze_model(self.outer_nerf.rgb_linear)

            # note: this is very important, otherwise the grid is all 1
            if self.cfg.nerf.sampler == "grid":
                print("Init grid based on pretrained outer nerf.")
                self.update_estimators(0)

        elif self.cfg.nerf.ckpt is not None and self.cfg.nerf.type == "nf":
            # load pretrained nerfacto as bg nerf
            print(f"==> Loading nerfacto from {self.cfg.nerf.ckpt}")
            param = torch.load(self.cfg.nerf.ckpt)
            nerfacto_config = ckpt2nsconfig(self.cfg.nerf.ckpt)["pipeline"]["model"]
            nerf_param = {k.replace("_model.", ""): v for k, v in param["pipeline"].items()}
            nerfacto_config = MyNerfactoModelConfig(**nerfacto_config)
            cam_embed_dim = nerf_param["field.embedding_appearance.embedding.weight"].shape[0]
            self.outer_nerf = NerfactoModel(nerfacto_config, kargs["scene_box"], cam_embed_dim)
            self.outer_nerf.load_state_dict(nerf_param)

            # nerfacto has to be fixed, as current training camera index does not match the pretrained one
            print("Freeze nerfacto.")
            freeze_model(self.outer_nerf)
            self.outer_nerf.eval()

        elif self.cfg.nerf.ckpt is not None:
            raise NotImplementedError

        elif self.cfg.nerf.type == "nep":
            nn.init.constant_(self.outer_nerf.rgb_linear.bias, np.log(0.5))

        self.color_network = AppShadingNetwork(
            self.cfg["shader_cfg"],
            lambda x: self.query_outer_nerf(x, self.cfg["shader_cfg"]["bg_sampler"])["rgb"],
        )
        self.sdf_inter_fun = lambda x: self.sdf_network.sdf(x)
        self.logged_mip = False

    def get_anneal_val(self, step):
        if self.cfg["anneal_end"] < 0:
            return 1.0
        else:
            return np.min([1.0, step / self.cfg["anneal_end"]])

    @staticmethod
    def near_far_from_sphere(rays_o, rays_d):
        """
        get near and far. In the following code, `mid` is the centeral point of two intersection points.
        """
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        near = torch.clamp(near, min=1e-3)
        return near, far

    def ref_score_wrapper(self, loss, extras):
        assert "ref_score" in extras
        assert (extras["ref_score"] >= 0).all()
        if self.cfg.score_weight_type == "div":
            weight = 1 / (extras["ref_score"] + 1e-4)

        elif self.cfg.score_weight_type == "neg":
            weight = 1 - extras["ref_score"]
            weight = weight.clamp(min=0.5, max=1.0)

        else:
            raise NotImplementedError

        weight = weight.clamp(min=0.0, max=self.cfg.score_weight_max)
        score_loss = loss * weight
        return torch.sum(score_loss, -1)

    def compute_rgb_loss(self, ray_rgb, rgb_gt, extras, is_coarse=False):
        # if is_coarse and self.cfg["rgb_loss"].startswith("score_"):
        #     type = self.cfg["rgb_loss"].replace("score_", "")
        # else:
        type = self.cfg["rgb_loss"]

        if type == "l2":
            rgb_loss = torch.sum((ray_rgb - rgb_gt) ** 2, -1)

        elif type == "l1":
            rgb_loss = torch.sum(F.l1_loss(ray_rgb, rgb_gt, reduction="none"), -1)

        elif type == "l2_l1":
            l2 = torch.sum((ray_rgb - rgb_gt) ** 2, -1)
            l1 = torch.sum(F.l1_loss(ray_rgb, rgb_gt, reduction="none"), -1)
            rgb_loss = 0.9 * l2 + 0.1 * l1

        elif type == "score_l1":
            l1 = F.l1_loss(ray_rgb, rgb_gt, reduction="none")
            return self.ref_score_wrapper(l1, extras)

        elif type == "score_l2":
            l2 = (ray_rgb - rgb_gt) ** 2
            return self.ref_score_wrapper(l2, extras)

        elif type == "score_l2_l1":
            l2 = (ray_rgb - rgb_gt) ** 2
            l1 = F.l1_loss(ray_rgb, rgb_gt, reduction="none")
            loss = 0.9 * l2 + 0.1 * l1
            return self.ref_score_wrapper(loss, extras)

        elif type == "smooth_l1":
            rgb_loss = torch.sum(F.smooth_l1_loss(ray_rgb, rgb_gt, reduction="none", beta=0.25), -1)

        elif type == "charbonier":
            epsilon = 0.001
            rgb_loss = torch.sqrt(torch.sum((rgb_gt - ray_rgb) ** 2, dim=-1) + epsilon)
        else:
            raise NotImplementedError
        return rgb_loss

    def density_activation(self, density, dists):
        return 1.0 - torch.exp(-F.softplus(density) * dists)

    @staticmethod
    def density2alpha(density, dists):
        return 1.0 - torch.exp(-density * dists)

    # def get_outer_density(self, points):
    #     points_norm = torch.norm(points, dim=-1, keepdim=True)
    #     points_norm = torch.clamp(points_norm, min=1e-3)

    #     # todo: here they use a trick: let the input of nerf be [0, 1] by feeding the direction of the input position and the magnitude of it dividedly.
    #     sigma = self.outer_nerf(
    #         torch.cat([points / points_norm, 1.0 / points_norm], -1),
    #         None,
    #         density_only=True,
    #     )[..., 0]
    #     return sigma

    @staticmethod
    def upsample(rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = next_z_vals - prev_z_vals
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = (
            alpha
            * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1.0 - alpha + 1e-7], -1), -1)[
                :, :-1
            ]
        )

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = (
                torch.arange(batch_size)[:, None]
                .expand(batch_size, n_samples + n_importance)
                .reshape(-1)
            )
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def nep_sample_bg_z_vals(self, batch_size, near, perturb, device):
        """
        Sample points from [near, +inf]
        more distant, less samples.
        """
        n_bg_samples = self.cfg["n_bg_samples"]
        z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (n_bg_samples + 1.0), n_bg_samples).to(
            device
        )
        if perturb > 0:
            # here the range of z_vals_outside is: [near, +inf]
            mids = 0.5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
            upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
            lower = torch.cat([z_vals_outside[..., :1], mids], -1)
            t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]]).to(device)
            z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        # most samples are around far. more distant, less samples.
        z_vals_outside = near / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / n_bg_samples
        return z_vals_outside

    def sample_ray(self, raybundle, perturb, step):
        rays_o, rays_d, near, far = (
            raybundle.origins,
            raybundle.directions,
            raybundle.nears,
            raybundle.fars,
        )
        n_samples = self.cfg["n_samples"]
        n_importance = self.cfg["n_importance"]
        up_sample_steps = self.cfg["up_sample_steps"]
        batch_size = len(rays_o)

        # sample points
        z_vals = torch.linspace(0.0, 1.0, n_samples).type_as(rays_o)  # sn
        z_vals = near + (far - near) * z_vals[None, :]  # rn,sn

        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5).type_as(rays_o)
            z_vals = z_vals + t_rand * 2.0 / n_samples

        # do improtance sampling based on the output of sdf network
        with torch.no_grad():
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]

            # in the case of mipnerf, points position is provided as gaussian mean. ie, cov = 0
            # todo: using proposal network to help sampling
            sdf = self.sdf_network.sdf(pts).reshape(batch_size, n_samples)

            for i in range(up_sample_steps):
                rn, sn = z_vals.shape
                if self.cfg["clip_sample_variance"]:
                    inv_s = self.deviation_network(torch.empty([1, 3])).expand(rn, sn - 1)
                    # prevent too large inv_s
                    inv_s = torch.clamp(inv_s, max=64 * 2**i)
                else:
                    inv_s = torch.ones(rn, sn - 1) * 64 * 2**i
                new_z_vals = self.upsample(
                    rays_o, rays_d, z_vals, sdf, n_importance // up_sample_steps, inv_s
                )
                z_vals, sdf = self.cat_z_vals(
                    rays_o,
                    rays_d,
                    z_vals,
                    new_z_vals,
                    sdf,
                    last=(i + 1 == up_sample_steps),
                )

        # z_vals_outside = self.nep_sample_bg_z_vals(batch_size, near, perturb, rays_o.device)
        # z_vals_outside = self.sample_ray_nerf(raybundle, sampler=self.cfg.nerf.sampler, step=step)
        # z_vals = torch.cat([z_vals, z_vals_outside], -1)
        return z_vals

    def forward_neus(
        self, xyz, dists, rays_d_pt, cos_anneal_ratio, step, human_poses_pt, inner_mask=...
    ):
        xyz = xyz[inner_mask]
        dists = dists[inner_mask]
        rays_d_pt = rays_d_pt[inner_mask]
        human_poses_pt = human_poses_pt[inner_mask]
        (inner_alpha, gradients, feature, inv_s, sdf) = self.get_neus_sdf_alpha(
            xyz, dists, rays_d_pt, cos_anneal_ratio, step
        )
        inner_rgb, extras = self.color_network(
            xyz, gradients, -rays_d_pt, feature, human_poses_pt, step=step
        )
        return inner_rgb, inner_alpha, extras, gradients, sdf, inv_s

    def ns_render(
        self,
        raybundle: RayBundle,
        perturb_overwrite=-1,
        cos_anneal=0.0,
        is_train=True,
        step: Optional[int] = None,
        log_vis=True,
    ):
        self.res = {"metrics": {}}
        res: Dict[str, Any] = self.res
        use_gaussian = self.use_mip
        assert self.cfg.nerf.sampler != "grid"
        use_acc = False
        assert step is not None
        n_rays = raybundle.shape[0]
        perturb = self.cfg["perturb"]
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        human_poses = raybundle.metadata["human_poses"].view(-1, 3, 4)

        # sample points for neus
        z_vals = self.sample_ray(raybundle, perturb, step)
        t_starts = z_vals[..., :-1]
        t_ends = z_vals[..., 1:]
        ray_indices = None

        # get nerf points and concat them
        near_bak, far_bak = raybundle.nears, raybundle.fars
        raybundle.nears, raybundle.fars = raybundle.fars, torch.full_like(
            raybundle.fars, self.cfg.nerf.far
        )
        nerf_samples = self.sample_ray_nerf(raybundle, sampler=self.cfg.nerf.sampler, step=step)

        # nerf_samples = self.samples_gs2acc(nerf_samples["ray_samples"])
        if self.cfg.nerf.sampler == "prop":
            res["weights_list"] = nerf_samples["weights_list"]
            res["ray_samples_list"] = nerf_samples["ray_samples_list"]

        nerf_samples = nerf_samples["ray_samples"]
        nerf_t_starts = nerf_samples.frustums.starts[..., 0]
        nerf_t_ends = nerf_samples.frustums.ends[..., 0]
        rays_o_pt = raybundle.origins[:, None, :]
        n_samples = t_starts.shape[1] + nerf_t_starts.shape[1]
        rays_d_pt = raybundle.directions.unsqueeze(-2).expand(n_rays, n_samples, 3)
        hp = human_poses.unsqueeze(-3).expand(n_rays, n_samples, 3, 4)

        t_starts = torch.cat([t_starts, nerf_t_starts], -1)
        t_ends = torch.cat([t_ends, nerf_t_ends], -1)

        # prepare variables
        raybundle.nears, raybundle.fars = near_bak, far_bak
        dists = t_ends - t_starts

        mid_z_vals = t_starts + dists * 0.5
        xyz = rays_o_pt + rays_d_pt * mid_z_vals.unsqueeze(-1)

        if use_gaussian and not use_acc:
            ray_samples = raybundle.get_ray_samples(
                bin_starts=t_starts[..., None], bin_ends=t_ends[..., None]
            )
            gaussians = ray_samples.frustums.get_gaussian_blob()
            mean, cov = gaussians.mean, gaussians.cov

        inner_mask = torch.norm(xyz, dim=-1) <= 1.0
        outer_mask = ~inner_mask

        rays_o_pt = F.normalize(rays_o_pt, dim=-1)
        alpha = torch.zeros(*t_starts.shape)
        rgbs, rgbs_nodir, Ls, grad_normals, pred_normals, ref_scores = [
            torch.zeros(*t_starts.shape, 3) for _ in range(6)
        ]

        # define functions
        def forward_nerf(mask):
            if use_gaussian:
                outer_points = mean[mask]
                outer_cov = cov[mask]
            elif self.cfg.nerf.type == "nf":
                ray_samples = raybundle.get_ray_samples(
                    bin_starts=t_starts[..., None], bin_ends=t_ends[..., None]
                )
                outer_points = ray_samples[mask]
                outer_cov = None
            else:
                outer_points = xyz[mask]
                outer_cov = None

            return self.outer_nerf_forward_samples(
                outer_points,
                dists[mask],
                -rays_d_pt[mask],
                cov=outer_cov,
            )

        # ===========================
        # Step 1. forward outer nerf
        # ===========================
        if torch.sum(outer_mask) > 0:
            # here the value range of nerfacto and neus may be different
            alpha[outer_mask], rgbs[outer_mask], nerf_res = forward_nerf(outer_mask)
            if self.cfg.nerf.use_refnerf:
                grad_normals[outer_mask] = nerf_res["grad_normal"]
                pred_normals[outer_mask] = nerf_res["pred_normal"]

        # ===========================
        # Step 2. forward inner neus
        # ===========================
        if torch.sum(inner_mask) > 0:
            # Step 2.1 - get neus sdf and alpha
            neus_res = self.forward_neus(xyz, dists, rays_d_pt, cos_anneal, step, hp, inner_mask)
            inner_rgb, inner_alpha, extras, gradients, sdf, inv_s = neus_res
            rgbs[inner_mask], alpha[inner_mask] = inner_rgb, inner_alpha
            inner_normals = F.normalize(gradients, dim=-1)

            # Eikonal loss
            gradient_error = (torch.linalg.norm(gradients, ord=2, dim=-1) - 1.0) ** 2

            if self.cfg.loss.normal_smooth > 0:
                normals = F.normalize(gradients, dim=-1).unsqueeze(1).expand(-1, 3, -1)
                n_offsets = 3
                inner_xyz = xyz[inner_mask]
                xyz_offset = (
                    inner_xyz.unsqueeze(1) + torch.randn(inner_xyz.shape[0], n_offsets, 3) * 3e-3
                )
                _, gradients = self.sdf_network(xyz_offset, with_grad=True)
                normals_offset = F.normalize(gradients, dim=-1)
                res["loss_normal"] = self.cfg.loss.normal_smooth * F.l1_loss(
                    normals_offset, normals
                )

            # Step 2.3 - extra losses
            if self.cfg.shader_cfg.use_refneus and "rgb_nodir" in extras:
                # if self.cfg.shader_cfg.use_refneus and self.cfg.rgb_loss.startswith("score_"):

                rgbs_nodir[inner_mask] = extras["rgb_nodir"]
                if self.cfg.shader_cfg.lr_decomp:
                    Ls[inner_mask] = extras["L"]

                if self.cfg["ref_score_detach"]:
                    rgb_nodir = extras["rgb_nodir"].detach()
                else:
                    rgb_nodir = extras["rgb_nodir"]

                if self.cfg.ref_score_type == "l2":
                    ref_scores[inner_mask] = F.mse_loss(
                        rgb_nodir, extras["rgb_withdir"], reduction="none"
                    )
                elif self.cfg.ref_score_type == "l1":
                    ref_scores[inner_mask] = F.l1_loss(
                        rgb_nodir, extras["rgb_withdir"], reduction="none"
                    )
                else:
                    raise NotImplementedError

            elif self.cfg["apply_occ_loss"]:
                extra_losses = self.compute_occ_loss(
                    extras, xyz[inner_mask], sdf, gradients, rays_d_pt[inner_mask], step
                )
                res["loss_occ"] = extra_losses[0].mean()
                if extra_losses[1] is not None:
                    # iol: inner occ light
                    res["loss_iol"] = extra_losses[1].mean()

            res["std"] = torch.mean(1 / inv_s)

            # curvature loss
            if self.cfg.loss.curvature.weight > 0:
                start = self.cfg.loss.curvature.reduce_start
                end = start + self.cfg.loss.curvature.reduce_step
                global_weight = map_range_val(step, start, end, 1, 0)
                _, curvature = self.get_curvature_loss(xyz[inner_mask], gradients)
                res["loss_cur"] = global_weight * self.cfg.loss.curvature.weight * curvature.mean()

        else:
            gradient_error = torch.zeros(1)
            res["loss_occ"] = torch.zeros(1).mean()
            res["std"] = torch.zeros(1)

        # =========================================
        # Step 3. volume rendering and get results
        # =========================================
        weights = (
            alpha
            * torch.cumprod(torch.cat([torch.ones([n_rays, 1]), 1.0 - alpha + 1e-7], -1), -1)[
                ..., :-1
            ]
        )
        vol_fn = lambda weights, value: (value * weights[..., None]).sum(dim=1)
        # weights, _ = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        # vol_fn = lambda weights, value: accumulate_along_rays(weights, value, ray_indices, n_rays)
        # res["ray_rgb"] = (rgbs * weights[..., None]).sum(dim=1)
        res["ray_rgb"] = vol_fn(weights, rgbs)
        res["gradient_error"] = gradient_error  # rn

        # orientation_loss
        if torch.sum(inner_mask) > 0 and self.cfg.loss.orientation > 0:
            weights_temp = weights[inner_mask].detach().clone()
            normals_temp = torch.zeros(*t_starts.shape, 3)
            normals_temp[inner_mask] = inner_normals
            res["loss_ori"] = (
                self.cfg.loss.orientation
                * orientation_loss(weights_temp, normals_temp, raybundle.directions).mean()
            )

        # for refneus
        if (
            self.cfg.shader_cfg.use_refneus
            and not self.cfg.shader_cfg.refneus_single_mlp
            and not self.cfg.shader_cfg.refneus_single_dir_mlp
        ):
            if not self.cfg.shader_cfg.lr_decomp:
                # when decomp lr, no need to supervised both final rgb and R
                rgbs_nodir[outer_mask] = rgbs[outer_mask]
                res["ray_rgb_coarse"] = vol_fn(weights, rgbs_nodir)
            elif not self.training:
                res["R"] = vol_fn(weights, rgbs_nodir)
                res["L"] = vol_fn(weights, Ls)

            res["ref_score"] = vol_fn(weights, ref_scores)

        if self.cfg.nerf.use_refnerf:
            res["loss_normal"] = pred_normal_loss(
                weights[..., None], grad_normals.detach(), pred_normals
            ).mean() * float(self.cfg.loss.pred_normal)

        if step < 1000:
            mask = torch.norm(xyz, dim=-1) < 1.2
            res["sdf_pts"] = xyz[mask]
            res["sdf_vals"] = self.sdf_network.sdf(xyz[mask])[..., 0]

        # vis
        if not is_train and log_vis:
            vis_dict = self.get_vis_data(
                raybundle, weights, t_starts, ray_indices, human_poses, step, use_acc=use_acc
            )
            # detach vis data
            for k, v in vis_dict.items():
                vis_dict[k] = v.detach()
            res.update(vis_dict)

        return res

    def get_inner_mask(self, points, bias=0, keepdim=False):
        """
        points: shape [N, 3]
        bias: postive for larger bounds, negative for smaller bounds
        """
        if self.cfg.bounds_type == "sphere":
            return torch.norm(points, dim=-1, keepdim=keepdim) <= 1.0 + bias

        elif self.cfg.bounds_type == "aabb":
            if self.min_pt.dtype != points.dtype:
                self.min_pt = self.min_pt.type_as(points)
                self.max_pt = self.max_pt.type_as(points)
            mask = (points >= self.min_pt + bias) & (points <= self.max_pt + bias)
            return mask.all(dim=-1, keepdim=keepdim)

        else:
            raise NotImplementedError

    def get_vis_data(
        self, raybundle, weights, t_starts, ray_indices, human_poses, step, use_acc=False
    ):
        # here weights num is 159 but z_vals num is 160:
        # if z_vals.shape[-1] == weights.shape[-1] + 1:
        #     z_vals = z_vals[..., :-1]

        # get the proterties for points in the depth plane

        n_rays = len(raybundle)
        rays_o, rays_d = raybundle.origins, raybundle.directions
        if use_acc:
            depth = accumulate_along_rays(weights, t_starts, ray_indices, n_rays)
        else:
            depth = torch.sum(weights * t_starts, -1, keepdim=True)  # rn,
        surface_xyz = depth * rays_d + rays_o  # rn,3

        feature, gradients = self.sdf_network(surface_xyz, with_grad=True)
        feature = feature[..., 1:]

        # feature_vector = self.sdf_network(points)[..., 1:]  # rn,f
        # gradients = self.sdf_network.gradient(points)  # rn,3

        inner_mask = self.get_inner_mask(surface_xyz, keepdim=True)
        outputs = {
            "depth": depth,  # rn,1
            # rn,3
            # "normal": ((F.normalize(gradients, dim=-1) + 1.0) * 0.5) * inner_mask,
            "normal": F.normalize(gradients, dim=-1) * inner_mask,
        }

        _, occ_info, inter_results = self.color_network(
            surface_xyz,
            gradients,
            -F.normalize(rays_d, dim=-1),
            feature,
            human_poses,
            inter_results=True,
            step=step,
        )
        # todo: use this func in stage 1
        _, occ_prob, _ = get_intersection(
            self.sdf_inter_fun,
            self.deviation_network,
            surface_xyz,
            occ_info["reflective"],
            sn0=128,
            sn1=9,
        )  # pn,sn-1
        occ_prob_gt = torch.sum(occ_prob, dim=-1, keepdim=True)
        outputs["occ_prob_gt"] = occ_prob_gt

        for k, v in inter_results.items():
            inter_results[k] = v * inner_mask
        outputs.update(inter_results)
        return outputs

    def get_neus_sdf_alpha(self, points, dists, dirs, cos_anneal_ratio, step):
        # points [...,3] dists [...] dirs[...,3]
        feature, gradients = self.sdf_network(points, with_grad=True)
        sdf = feature[..., 0]
        feature = feature[..., 1:]
        # gradients = self.sdf_network.gradient(points)  # ...,3

        # todo: should we use a single global leanable inv_s or per-point inv_s?
        inv_s = self.deviation_network(points).clip(1e-6, 1e6)  # ...,1
        inv_s = inv_s[..., 0]

        if self.cfg["freeze_inv_s_step"] is not None and step < self.cfg["freeze_inv_s_step"]:
            inv_s = inv_s.detach()

        if self.cfg.unbiased_sigma:
            # todo: should we detach the normal here?
            normal = F.normalize(gradients, p=2, dim=-1).detach()
            true_cos = (dirs * normal).sum(-1)  # [...]
            density = inv_s * torch.sigmoid(-inv_s * sdf / (0.999 * true_cos.abs() + 0.001))

            # cos_anneal_ratio = min(cos_anneal_ratio, 1)
            # iter_cos = (1 - cos_anneal_ratio) + cos_anneal_ratio * true_cos.abs()
            # density = inv_s * torch.sigmoid(-inv_s * sdf / iter_cos)
            return (self.density2alpha(density, dists).clip(0, 1), gradients, feature, inv_s, sdf)

        true_cos = (dirs * gradients).sum(-1)  # [...]
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
            + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)  # [...]
        # check_memory_usage('neus_fn: after getting alpha')
        return alpha, gradients, feature, inv_s, sdf

    def outer_nerf_forward_samples(
        self, samples, dists, dirs, cov=None, normals=None, return_linear=False
    ):
        # note: here dirs are flipped !!!
        if self.cfg.nerf.type == "nf":
            field_outputs = self.outer_nerf.field(samples, compute_normals=False)
            return (
                field_outputs[FieldHeadNames.DENSITY].squeeze(-1),
                field_outputs[FieldHeadNames.RGB],
                field_outputs,
            )

        res = self.outer_nerf(samples, dirs, cov=cov, normals=normals)
        density, color = res["density"], res["rgb"]

        alpha = self.density_activation(density[..., 0], dists)
        if self.cfg.nerf.rgb_exp:
            color = torch.exp(torch.clamp(color, max=5.0))
        else:
            color = F.sigmoid(color)

        if not return_linear:
            color = linear_to_srgb(color)

        return alpha, color, res

    def sample_ray_nerf(self, ray_bundle, sampler="uniform", step=None):
        if self.cfg.nerf.type == "nf":
            ray_samples, weights_list, ray_samples_list = self.outer_nerf.proposal_sampler(
                ray_bundle, density_fns=self.outer_nerf.density_fns
            )
            return {"ray_samples": ray_samples}

        assert sampler in ["uniform", "pdf", "grid", "nep_bg", "prop"]
        # if sampler == "grid":
        #     t_starts, t_ends, ray_indices = self.grid_sampling_nerf(
        #         ray_bundle,
        #         self.cfg.nerf.near,
        #         self.cfg.nerf.far,
        #     )
        #     return t_starts, t_ends, ray_indices

        if sampler == "nep_bg":
            z_vals = self.nep_sample_bg_z_vals(
                ray_bundle.shape[0],
                ray_bundle.nears,
                self.cfg["perturb"],
                self.device,
            )
            ray_samples = ray_bundle.get_ray_samples(
                bin_starts=z_vals[..., :-1, None],
                bin_ends=z_vals[..., 1:, None],
            )

        elif sampler == "uniform":
            # sample using rays' near far
            ray_samples_uniform = self.bg_sampler_uniform(ray_bundle)
            ray_samples = ray_samples_uniform

        elif sampler == "pdf":
            # sample using rays' near far
            ray_samples_uniform = self.bg_sampler_uniform(ray_bundle)
            if self.use_mip:
                coarse_gaussians = ray_samples_uniform.frustums.get_gaussian_blob()
                mean, cov = coarse_gaussians.mean, coarse_gaussians.cov
            else:
                mean = ray_samples_uniform.frustums.get_positions()
                cov = None

            res = self.outer_nerf(mean, -ray_samples_uniform.frustums.directions, cov)
            weights_coarse = ray_samples_uniform.get_weights(res["density"])
            ray_samples = self.bg_sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        elif sampler == "prop":
            # assert step is not None

            # def prop_sigma_fn(t_starts, t_ends, proposal_network):
            #     gaussians = ray_bundle.get_ray_samples(
            #         bin_starts=t_starts[..., None], bin_ends=t_ends[..., None]
            #     ).frustums.get_gaussian_blob()
            #     sigmas = proposal_network(gaussians.mean, None, gaussians.cov, density_only=True)[
            #         "density"
            #     ]
            #     return sigmas.squeeze(-1)

            # option 1: use nerfacc
            # the num _samples of prop net is more than final nerf
            # requires_grad = self.proposal_requires_grad_fn(step)
            # with torch.set_grad_enabled(requires_grad):
            #     t_starts, t_ends = self.prop_sampler.sampling(
            #         prop_sigma_fns=[
            #             lambda *args: prop_sigma_fn(*args, p) for p in self.proposal_networks
            #         ],
            #         prop_samples=[self.cfg["n_bg_samples_importance"]]
            #         * len(self.proposal_networks),
            #         num_samples=self.cfg["n_bg_samples"],
            #         n_rays=ray_bundle.shape[0],
            #         near_plane=ray_bundle.nears,
            #         far_plane=ray_bundle.fars,
            #         stratified=self.outer_nerf.training,
            #         requires_grad=requires_grad,
            #     )
            # ray_samples = ray_bundle.get_ray_samples(
            #     bin_starts=t_starts[..., None], bin_ends=t_ends[..., None]
            # )

            # option 2: use ns
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
                ray_bundle, density_fns=self.density_fns, use_gaussian=self.use_mip
            )
            return {
                "ray_samples": ray_samples,
                "weights_list": weights_list,
                "ray_samples_list": ray_samples_list,
            }

        else:
            raise NotImplementedError
        return {"ray_samples": ray_samples}

    def query_outer_nerf(
        self,
        ray_bundle: RayBundle,
        sampler="uniform",
        returns=["rgb"],
        step=None,
        return_linear=False,
    ):
        """
        Given a set of rays, return rgb, depth.
        sample and render
        """
        assert ray_bundle.nears is not None and ray_bundle.fars is not None
        if self.cfg.nerf.type == "nf":
            return self.outer_nerf.get_outputs(ray_bundle)

        sampler_res = self.sample_ray_nerf(
            ray_bundle,
            sampler=sampler,
            step=step,
        )

        if sampler != "grid":
            ray_samples: RaySamples = sampler_res["ray_samples"]
            dirs = ray_samples.frustums.directions

            # samples to mean and cov
            if self.use_mip:
                gaussian_samples = ray_samples.frustums.get_gaussian_blob()
                mean, cov = gaussian_samples.mean, gaussian_samples.cov
            else:
                mean = ray_samples.frustums.get_positions()
                cov = None

            dists = ray_samples.frustums.ends - ray_samples.frustums.starts
            dists = dists.squeeze(-1)
            alpha, rgb, _ = self.outer_nerf_forward_samples(
                mean, dists, -dirs, cov=cov, return_linear=return_linear
            )
            weights, trans = render_weight_from_alpha(alpha, ray_indices=None, n_rays=None)
            res = {}
            res["weights"] = weights
            res["trans"] = trans
            if "rgb" in returns:
                res["rgb"] = (rgb * weights[..., None]).sum(dim=1)
            if "depth" in returns:
                res["depth"] = torch.sum(
                    weights[..., None] * ray_samples.frustums.starts, -2
                )  # rn,

            # update prop net
            if self.cfg.nerf.sampler == "prop" and self.outer_nerf.training:
                #     with torch.set_grad_enabled(self.training):
                #         self.prop_sampler.update_every_n_steps(
                #             res["trans"], self.proposal_requires_grad_fn(step), loss_scaler=1
                #         )
                res["weights_list"] = sampler_res["weights_list"]
                res["ray_samples_list"] = sampler_res["ray_samples_list"]

            return res

        else:
            t_starts, t_ends, ray_indices = sampler_res
            outputs = self.grid_rendering_nerf(ray_bundle, t_starts, t_ends, ray_indices)
            # outputs["rgb"] = outputs.pop("ray_rgb")
            return outputs

    def get_curvature_loss(self, points, sdf_gradients):
        # get the curvature along a certain random direction for each point
        # does it by computing the normal at a shifted point on the tangent plant and then computing a dot produt

        # to the original positions, add also a tiny epsilon
        # nr_points_original = points.shape[0]
        epsilon = 1e-4
        rand_directions = torch.randn_like(points)
        rand_directions = F.normalize(rand_directions, dim=-1)

        # instead of random direction we take the normals at these points, and calculate a random vector that is orthogonal
        normals = F.normalize(sdf_gradients, dim=-1)
        # normals=normals.detach()
        tangent = torch.cross(normals, rand_directions)
        rand_directions = tangent  # set the random moving direction to be the tangent direction now

        points_shifted = points.clone() + rand_directions * epsilon

        # get the gradient at the shifted point
        feature, sdf_gradients_shifted = self.sdf_network(points_shifted, with_grad=True)
        sdf_shifted = feature[..., 0]
        feature = feature[..., 1:]

        normals_shifted = F.normalize(sdf_gradients_shifted, dim=-1)
        dot = (normals * normals_shifted).sum(dim=-1, keepdim=True)
        # the dot would assign low weight importance to normals that are almost the same, and increasing error the more they deviate. So it's something like and L2 loss. But we want a L1 loss so we get the angle, and then we map it to range [0,1]
        angle = torch.acos(
            torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6)
        )  # goes to range 0 when the angle is the same and pi when is opposite

        curvature = angle / math.pi  # map to [0,1 range]
        return sdf_shifted, curvature

    def compute_occ_loss(self, occ_info, points, sdf, gradients, dirs, step):
        if step < self.cfg["occ_loss_step"]:
            return torch.zeros(1), None

        occ_prob = occ_info["occ_prob"]
        reflective = occ_info["reflective"]

        # select a subset for occ loss
        # note we only apply occ loss on the surface
        inner_mask = self.get_inner_mask(points, bias=-0.001)
        sdf_mask = torch.abs(sdf) < self.cfg["occ_sdf_thresh"]
        normal_mask = torch.sum(gradients * dirs, -1) < 0  # pn
        mask = inner_mask & normal_mask & sdf_mask

        if torch.sum(mask) > self.cfg["occ_loss_max_pn"]:
            indices = torch.nonzero(mask)[:, 0]  # npn
            idx = torch.randperm(indices.shape[0], device="cuda")  # npn
            indices = indices[idx[: self.cfg["occ_loss_max_pn"]]]  # max_pn
            mask_new = torch.zeros_like(mask)
            mask_new[indices] = 1
            mask = mask_new

        inner_loss = None
        if torch.sum(mask) > 0:
            inter_dist, inter_prob, inter_sdf = get_intersection(
                self.sdf_inter_fun,
                self.deviation_network,
                points[mask],
                reflective[mask],
                sn0=64,
                sn1=16,
            )  # pn,sn-1
            occ_prob_gt = torch.sum(inter_prob, -1, keepdim=True)
            occ_loss = F.l1_loss(occ_prob[mask], occ_prob_gt)

            # inner light loss
            if self.cfg.loss.inner.weight > 0:
                depth = torch.sum(inter_dist * inter_prob, -1, keepdim=True)
                second_rays_d = reflective[mask]
                inter_xyz = points[mask] + depth * second_rays_d
                feature, normals = self.sdf_network(inter_xyz, with_grad=True)
                sdf = feature[..., 0]
                feature = feature[..., 1:]
                inter_rgb, _ = self.color_network(
                    inter_xyz, normals, -second_rays_d, feature, None, step=step
                )
                weights = occ_prob_gt.detach() if self.cfg.loss.inner.weight_detach else occ_prob_gt
                gt_rgb = (
                    occ_info["inner_light"][mask].detach()
                    if self.cfg.loss.inner.gt_detach
                    else occ_info["inner_light"][mask]
                )
                inner_loss = F.l1_loss(inter_rgb, gt_rgb) * weights * self.cfg.loss.inner.weight
        else:
            occ_loss = torch.zeros(1)

        return occ_loss, inner_loss

    def predict_materials(self):
        name = self.cfg["name"]
        mesh = open3d.io.read_triangle_mesh(f"data/meshes/{name}-300000.ply")
        xyz = np.asarray(mesh.vertices)
        xyz = torch.from_numpy(xyz.astype(np.float32)).cuda()
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        metallic, roughness, albedo = [], [], []
        batch_size = 8192
        for vi in range(0, xyz.shape[0], batch_size):
            feature_vectors = self.sdf_network(xyz[vi : vi + batch_size])[:, 1:]
            m, r, a = self.color_network.predict_materials(
                xyz[vi : vi + batch_size], feature_vectors
            )
            metallic.append(m.cpu().numpy())
            roughness.append(r.cpu().numpy())
            albedo.append(a.cpu().numpy())

        return {
            "metallic": np.concatenate(metallic, 0),
            "roughness": np.concatenate(roughness, 0),
            "albedo": np.concatenate(albedo, 0),
        }

    # # nerfacc
    # def grid_render_neus(
    #     self,
    #     raybundle: RayBundle,
    #     perturb_overwrite=-1,
    #     cos_anneal_ratio=0.0,
    #     is_train=True,
    #     step: Optional[int] = None,
    # ):
    #     """
    #     render rgb using occ grid sampling with two estimators or single estimator
    #     """

    #     assert hasattr(self, "estimator")
    #     if self.training:
    #         self.update_estimators(step)

    #     use_gaussian = self.use_mip
    #     assert step is not None
    #     self.show_mip_info(use_gaussian)
    #     human_poses = raybundle.metadata["human_poses"].view(-1, 3, 4)

    #     # sample rays
    #     t_starts, t_ends, ray_indices = self.occ_grid_sampling_nerf(raybundle)
    #     res = self.occ_grid_rendering_all(
    #         raybundle, t_starts, t_ends, ray_indices, human_poses, cos_anneal_ratio, step
    #     )

    #     # vis
    #     # if not is_train:
    #     #     vis_dict = self.get_vis_data(z_vals, rays_o, rays_d, weights, human_poses, step)
    #     #     res.update(vis_dict)

    #     return res

    # def update_estimators(self, step):
    #     assert self.use_mip

    #     def occ_eval_fn_nerf(x):
    #         # todo: modify cov here.
    #         density = self.outer_nerf(
    #             x,
    #             None,
    #             # cov=(torch.eye(3).to(x) * 1e-1).expand(x.shape[0], 3, 3),
    #             cov=None,
    #             density_only=True,
    #         )["density"]
    #         alpha = density * 1e-3
    #         # alpha = density * 1
    #         # print("occ:", alpha.max(), alpha.min(), alpha.mean())
    #         return alpha

    #     # def occ_eval_fn_neus(x):
    #     #     ...

    #     if self.estimator_nerf is not None:
    #         self.estimator_nerf.update_every_n_steps(
    #             step=step, occ_eval_fn=occ_eval_fn_nerf, occ_thre=0.01
    #         )
    #     # if self.estimator_neus is not None:
    #     #     raise NotImplementedError
    #     #     self.estimator_neus.update_every_n_steps(
    #     #         step=step, occ_eval_fn=occ_eval_fn_neus, occ_thre=0.01
    #     #     )

    @staticmethod
    def samples_acc2gs(ray_bundle, t_starts, t_ends, ray_indices, return_dir=False):
        samples_raybundle = ray_bundle[ray_indices]
        frustums = Frustums(
            origins=samples_raybundle.origins,
            directions=samples_raybundle.directions,  # [..., 1, 3]
            starts=t_starts[..., None],  # [..., num_samples, 1]
            ends=t_ends[..., None],  # [..., num_samples, 1]
            pixel_area=samples_raybundle.pixel_area,
        )
        gaussians = frustums.get_gaussian_blob()
        if return_dir:
            return gaussians.mean, gaussians.cov, samples_raybundle.directions
        else:
            return gaussians.mean, gaussians.cov

    @staticmethod
    def samples_points2acc(t_starts, t_ends):
        ray_indices = (
            torch.arange(t_starts.shape[0])[:, None].expand(-1, t_starts.shape[1]).flatten()
        )
        return t_starts.flatten(), t_ends.flatten(), ray_indices

    @staticmethod
    def samples_gs2acc(ray_samples):
        n_rays, n_samples = ray_samples.shape
        t_starts = ray_samples.frustums.starts.flatten()
        t_ends = ray_samples.frustums.ends.flatten()
        ray_indices = torch.arange(n_rays)[:, None].expand(-1, n_samples).flatten()
        return t_starts, t_ends, ray_indices

    # def grid_sampling_nerf(self, ray_bundle: RayBundle, near=0.2, far=1e3):
    #     assert self.use_mip

    #     def sampling_fn(t_starts: Tensor, t_ends: Tensor, ray_indices: Tensor) -> Tensor:
    #         mean, cov = self.samples_acc2gs(ray_bundle, t_starts, t_ends, ray_indices)
    #         density = self.outer_nerf(
    #             mean,
    #             None,
    #             cov,
    #             density_only=True,
    #         )["density"]
    #         # print("sampling all:", density.shape[0])
    #         # if density.shape[0]:
    #         #     print("sampling:", density.max(), density.min(), density.mean())
    #         # return density[..., 0]
    #         alpha = self.density_activation(density[..., 0], t_ends - t_starts)
    #         return alpha

    #     with torch.no_grad():
    #         ray_indices, t_starts, t_ends = self.estimator_nerf.sampling(
    #             ray_bundle.origins,
    #             ray_bundle.directions,
    #             # sigma_fn=sampling_fn,
    #             alpha_fn=sampling_fn,
    #             stratified=self.training,
    #             near_plane=near,
    #             far_plane=far,
    #             t_min=ray_bundle.nears,
    #             t_max=ray_bundle.fars,
    #             render_step_size=1e-3,
    #             cone_angle=1e-3,
    #             alpha_thre=1e-2,
    #         )
    #         # print("sampling out:", len(ray_indices))
    #     return t_starts, t_ends, ray_indices

    # def grid_rendering_nerf(
    #     self,
    #     raybundle,
    #     t_starts,
    #     t_ends,
    #     ray_indices,
    # ):
    #     outputs = {}

    #     def rendering_fn(
    #         t_starts: Tensor, t_ends: Tensor, ray_indices: Tensor
    #     ) -> Tuple[Tensor, Tensor]:
    #         mean, cov, dirs = self.samples_acc2gs(raybundle, t_starts, t_ends, ray_indices, True)
    #         dists = t_ends - t_starts
    #         alpha, rgb, _ = self.outer_nerf_forward_samples(mean, dists, -dirs, cov)
    #         return rgb, alpha

    #     rgb, accumulation, depth, extras = nerfacc.rendering(
    #         t_starts,
    #         t_ends,
    #         ray_indices,
    #         n_rays=raybundle.origins.shape[0],
    #         # rgb_sigma_fn=rendering_fn,
    #         rgb_alpha_fn=rendering_fn,
    #         render_bkgd=torch.tensor([1, 1, 1]).to(t_starts),
    #     )
    #     outputs["rgb"] = rgb
    #     outputs["depth"] = depth
    #     outputs["opacity"] = accumulation
    #     outputs["weights"] = extras["weights"]
    #     outputs["trans"] = extras["trans"]
    #     return outputs
