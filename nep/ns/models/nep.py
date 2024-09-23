from __future__ import annotations
from nerfstudio.model_components.losses import interlevel_loss
import re
from nerfstudio.utils import colormaps
from nerfstudio.data.scene_box import SceneBox
from nep.config import get_config
import time
from tqdm import tqdm

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type, Optional, Union
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
import torch
from torch.nn import Parameter
from collections import defaultdict

from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.configs.config_utils import to_immutable_dict

from nep.network.renderers.shape import NePShapeRenderer
from nep.network.renderers.material import NePMaterialRenderer
from nep.network.renderers import name2renderer
from nep.network.loss import name2loss
from nep.utils.base_utils import normalize_using_percentile
from nep.utils.draw_utils import add_text_to_image
import torchvision.utils as vutils
import math
from nep.network.renderers.nerf import NeRFRenderer
from nep.network.renderers.neilf import NeILFModel


def make_image_grid(tensor):
    nrow = int(math.sqrt(tensor.shape[0]))
    return vutils.make_grid(tensor.permute(0, 3, 1, 2), nrow=nrow).permute(1, 2, 0)


@dataclass
class NePModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: NePModel)
    nep_cfg: Optional[str] = None  # input of get_config
    in_viewer: bool = False
    vis_mat: bool = False


def parse_arch_from_nep_str(nep_cfg):
    match_res = re.search(r"-m\s+(\S+)", nep_cfg)
    arch = match_res.group(1) if match_res else None
    if arch is None:
        raise RuntimeError(f"Can not parse arch from {nep_cfg}")
    return arch


class NePModel(Model):
    config: NePModelConfig

    def __init__(
        self,
        config: ModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        trace_fn=None,
        **kwargs,
    ) -> None:
        self.trace_fn = trace_fn
        self.colormap_options = colormaps.ColormapOptions(colormap="turbo")
        super().__init__(config, scene_box, num_train_data, **kwargs)

    def populate_modules(self):
        # assert self.config.arch is not None
        self.nep_config: Dict = get_config(command_string=self.config.nep_cfg, root="./nep")

        self.arch = parse_arch_from_nep_str(self.config.nep_cfg)
        print(f"Model arch: {self.arch}")

        # self.nep_config['mesh'] = self.config.mesh
        modelclass = self.nep_config.pop("network")
        self.network = name2renderer[modelclass](
            self.nep_config,
            self.trace_fn,
            scene_box=self.scene_box,
            num_train_data=self.num_train_data,
        )
        if modelclass == "material":
            assert self.trace_fn is not None

        # for neilf network:
        if isinstance(self.network, NeILFModel):
            self.init_neilf_loss_fn(self.network.cfg)

        print("Initialize network:", type(self.network))
        self.train_step = 0

        self.loss_fns = []
        for loss_name in self.nep_config["loss"]["ex"]:
            self.loss_fns.append(name2loss[loss_name](self.nep_config))

        # self.vis_keys = [
        #     "ray_rgb",
        #     "depth",
        #     "normal",
        #     "metallic",
        #     "roughness",
        #     "diffuse_albedo",
        #     "diffuse_lights",
        #     "diffuse_colors",
        #     "specular_albedo",
        #     "specular_lights",
        #     "specular_weights",
        #     "specular_colors",
        #     "specular_ref",
        #     "occ_prob",
        #     "indirect_light",
        #     "human_light",
        #     "plenoptic_rgb",
        #     "plenoptic_depth",
        #     "plenoptic_density_normal",
        #     "plenoptic_pred_normal",
        # ]
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        res = {}
        params = list(self.network.parameters())
        res["network"] = params
        if hasattr(self.network, "proposal_networks"):
            res["prop"] = list(self.network.proposal_networks.parameters())
        return res

    def get_outputs(self, ray_bundle: RayBundle, is_train: bool = True) -> Dict[str, torch.Tensor]:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        # get near far
        # for stage 1 only, as stage 2 will not use near and far
        if isinstance(self.network, NePShapeRenderer) and not isinstance(
            self.network, NeRFRenderer
        ):
            ray_bundle.nears, ray_bundle.fars = NePShapeRenderer.near_far_from_sphere(
                ray_bundle.origins, ray_bundle.directions
            )

        # get args for ns_render
        step = self.train_step
        if is_train:
            if hasattr(self.network, "get_anneal_val"):
                perturb_overrite = -1
                anneal = self.network.get_anneal_val(step)
            else:
                perturb_overrite = anneal = 0
        else:
            perturb_overrite = anneal = 0

        outputs = self.network.ns_render(
            ray_bundle, perturb_overrite, anneal, is_train=is_train, step=step
        )
        torch.set_default_tensor_type("torch.FloatTensor")
        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        torch.cuda.empty_cache()
        start_time = time.time()
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            # check_memory_usage(f'chunk {i} / {num_rays} | before forward')

            # fetch a chunk
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)

            # filter out bg rays if given mesh
            outputs = {}
            if self.trace_fn is not None:
                inters, normals, depth, hit_mask = self.trace_fn(
                    ray_bundle.origins, ray_bundle.directions
                )
                hit_mask = hit_mask[:, 0].to(ray_bundle.origins.device)
                hit_ray_bundle = ray_bundle[hit_mask]
                hit_ray_bundle.metadata.update(
                    {
                        "normals": normals[hit_mask],
                        "inters": inters[hit_mask],
                        "depth": depth[hit_mask],
                    }
                )
                obj_outputs = self(hit_ray_bundle, is_train=False)
                outputs["hit_mask"] = hit_mask
                # outputs["roughness_sqrt"] = torch.zeros([hit_mask.shape[0], 1]).to(
                #     hit_ray_bundle.origins.device
                # )
                for k in obj_outputs.keys():
                    if k in ["human_light", "metrics"]:
                        continue
                    if self.config.in_viewer and k == "ray_rgb":
                        continue

                    outputs[k] = torch.zeros(inters.shape[0], obj_outputs[k].shape[-1]).to(
                        hit_ray_bundle.origins.device
                    )
                    if obj_outputs[k].shape[0] == 0:
                        continue

                    # if "roughness" == k:
                    #     outputs["roughness_sqrt"][hit_mask] = torch.sqrt(obj_outputs[k])
                    outputs[k][hit_mask] = obj_outputs[k]

                outputs["normal"] = normals
                outputs["normal"] = outputs["normal"] * 0.5 + 0.5

            else:
                outputs = self(ray_bundle, is_train=False)

            # vis nerf
            near, far = NePShapeRenderer.near_far_from_sphere(
                ray_bundle.origins, ray_bundle.directions
            )
            # self.vis_plenoptic(ray_bundle, outputs, near=near, far=far, suffix="_inner")
            self.vis_plenoptic(ray_bundle, outputs, near=far, far=1e3, suffix="_outer")
            self.vis_plenoptic(ray_bundle, outputs, near=0.1, far=1e3, suffix="_all")
            # self.vis_plenoptic(ray_bundle, outputs, near=3.0, far=1e3, suffix="_far")

            for output_name, output in outputs.items():  # type: ignore
                # if not torch.is_tensor(output) or not self.is_vis_key(output_name):
                if (
                    not torch.is_tensor(output)
                    or output_name.startswith("loss_")
                    or output_name in ["gradient_error", "std", "sdf_pts", "sdf_vals"]
                ):
                    continue

                # add the cpu version tensor to avoid cuda OOM
                outputs_lists[output_name].append(output.detach().cpu())
                del output

        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(
                image_height, image_width, -1
            )  # type: ignore

        tqdm.write(
            f"Done eval image raybundle: {image_height}x{image_width}, {time.time() - start_time:.2f}s"
        )
        return outputs

    def vis_plenoptic(self, ray_bundle, outputs, near, far, suffix):
        bg_model = None
        if isinstance(self.network, NePShapeRenderer) and not isinstance(
            self.network, NeRFRenderer
        ):
            bg_model = self.network
            sampler = self.network.cfg["nerf"]["sampler"]

        elif isinstance(self.network, NePMaterialRenderer) and self.network.bg_model is not None:
            bg_model = self.network.shader_network.bg_model
            sampler = self.network.shader_network.cfg["nerf_sampler"]

        if bg_model is not None:
            if isinstance(near, torch.Tensor) and isinstance(far, torch.Tensor):
                ray_bundle.nears, ray_bundle.fars = near, far
            else:
                bs = ray_bundle.origins.shape[0]
                ray_bundle.nears, ray_bundle.fars = (
                    torch.ones((bs, 1)).to(ray_bundle.origins.device) * near,
                    torch.ones((bs, 1)).to(ray_bundle.origins.device) * far,
                )
            nerf_res = bg_model.query_outer_nerf(
                ray_bundle,
                sampler=sampler,
                returns=["rgb", "depth"],
            )
            outputs.update(
                {f"plenoptic_{k}{suffix}-{sampler}": nerf_res[k] for k in ["rgb", "depth"]}
            )

    # def is_vis_key(self, k):
    #     for x in self.vis_keys:
    #         if k.startswith(x):
    #             return True
    #     return False

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_img = batch["image"].to(outputs["ray_rgb"].device)[..., :3]
        rgb_fine = outputs["ray_rgb"]

        # normalize depth
        for k in outputs:
            if "depth" in k:
                outputs[k] = normalize_using_percentile(outputs[k])
            # if "normal" in k:
            #     outputs[k] = outputs[k] * 0.5 + 0.5

        # add mock gt reference for syn data
        if "roughness" in outputs:
            outputs["1.0"] = torch.ones_like(outputs["roughness"])
            outputs["1.0"][outputs["roughness"] == 0] = 0
            outputs["0.1"] = torch.full_like(outputs["roughness"], 0.1)
            outputs["0.1"][outputs["roughness"] == 0] = 0

        # for paper comparison
        images_dict = {}
        # for mat_key in [
        #     "roughness",
        #     "metallic",
        #     "GT",
        #     "ray_rgb",
        #     "diffuse_color",
        #     "specular_color",
        # ]:
        #     if mat_key in outputs:
        #         images_dict[mat_key] = outputs[mat_key]
        #         # images_dict[mat_key][outputs["roughness"] == 0] = 1

        if self.config.vis_mat:
            names = "ray_rgb normal roughness squared_roughness real_roughness metallic albedo diffuse_color specular_color approximate_light hit_mask"
            names = names.split(" ")
            text_fn = lambda x, text: x
            colormap_fn = lambda x: x.expand_as(gt_img)
            make_grid_fn = lambda x: torch.cat([y for y in x], dim=1)
            key_source = names
        else:
            text_fn = add_text_to_image
            colormap_fn = lambda x: colormaps.apply_colormap(x, self.colormap_options)
            make_grid_fn = make_image_grid
            key_source = outputs

        combined_vis_img = torch.stack(
            [
                text_fn(gt_img, f"GT-step{self.train_step}"),
                *[
                    text_fn(outputs[x], x)
                    if outputs[x].shape[-1] == 3
                    else text_fn(colormap_fn(outputs[x]), x)
                    for x in key_source
                ],
            ],
            dim=0,
        )

        # add an alpha channel for combined_vis_img
        alpha = torch.ones_like(combined_vis_img[:, :, :, :1])

        gt_img = torch.moveaxis(gt_img, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]
        rgb_fine = torch.clip(rgb_fine, min=0, max=1)
        fine_psnr = self.psnr(gt_img, rgb_fine)
        fine_ssim = self.ssim(gt_img, rgb_fine)

        assert isinstance(fine_ssim, torch.Tensor)
        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "ssim": float(fine_ssim.item()),
        }
        # warn: in stage 2, here the pred image containes object with black background, which will cause very low psnr (~7)
        tqdm.write(
            f'Calc metrics {self.train_step}: psnr={metrics_dict["psnr"]:.4f}, ssim={metrics_dict["ssim"]:.4f}'
        )
        images_dict[f"vis-im{batch['image_idx']}"] = make_grid_fn(combined_vis_img)
        # "rgb": rgb_fine.permute(0, 2, 3, 1)[0],
        return metrics_dict, images_dict

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_train_step(step):
            self.train_step += 1

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_train_step,
            )
        ]

    def init_neilf_loss_fn(self, config):
        rgb_loss_type = config["train"]["rgb_loss"]
        lambertian_weighting = config["train"]["lambertian_weighting"]
        smoothness_weighting = config["train"]["smoothness_weighting"]
        trace_weighting = config["train"]["trace_weighting"]
        var_weighting = config["train"]["var_weighting"]
        remove_black = config["train"].get("remove_black", False)
        if config["geometry_module"] == "model.geo_fixmesh":
            rf_output_scale = config["geometry"]["slf_network"].get("sigmoid_output_scale", 1.0)
        elif config["geometry_module"] == "model.geo_volsdf":
            rf_output_scale = config["geometry"]["rendering_network"].get(
                "sigmoid_output_scale", 1.0
            )
        from nep.network.renderers.neilf_utils.geo_volsdf import GeoLoss
        from nep.network.renderers.neilf_utils.loss import NeILFLoss

        geo_loss = GeoLoss(rf_loss_scale_mod=1 / rf_output_scale, **config["train"]["geo_loss"])
        self.neilf_loss = NeILFLoss(
            rgb_loss_type,
            lambertian_weighting,
            smoothness_weighting,
            trace_weighting,
            var_weighting,
            geo_loss,
            self.network.phase,
            mono_neilf=config["use_ldr_image"],
            remove_black=remove_black,
            trace_grad_scale=config["train"].get("trace_grad_scale", 0),
        )

    def get_neilf_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        batch["image"] = batch["image"].type_as(outputs["ray_rgb"])[..., :3]
        batch["rgb"] = batch["image"]
        # return {"loss_neilf": self.neilf_loss(outputs, batch, self.train_step)}
        return self.neilf_loss(outputs, batch, self.train_step)

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        if isinstance(self.network, NeILFModel):
            return self.get_neilf_loss_dict(outputs, batch, metrics_dict)

        # get main loss
        batch["image"] = batch["image"].type_as(outputs["ray_rgb"])[..., :3]
        outputs["loss_rgb"] = self.network.compute_rgb_loss(
            outputs["ray_rgb"], batch["image"], outputs
        )

        # get coarse rgb loss (for mip0 or ref-neus)
        # if "coarse_rgb_weight" in self.nep_config and self.nep_config["coarse_rgb_weight"] > 0:
        if "ray_rgb_coarse" in outputs:
            loss_dict["loss_rgb_coarse"] = (
                self.network.compute_rgb_loss(
                    outputs["ray_rgb_coarse"], batch["image"], outputs, is_coarse=True
                )
                * self.nep_config["coarse_rgb_weight"]
            ).mean()

        # prop net loss
        if "weights_list" in outputs:
            loss_dict["loss_prop"] = interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )

        # extra losses: bake loss, normal loss ...
        for k, v in outputs.items():
            if k.startswith("loss_") and v is not None:
                loss_dict[k] = v

        # calc and collect other losses
        for func in self.loss_fns:
            # this `func` could be a function to compute loss from outputs, or just to fetch loss_xxx from outputs.
            # here param `data_gt` is not used.
            loss_results = func(outputs, batch, self.train_step)
            for k, v in loss_results.items():
                if k.startswith("loss") and v is not None:
                    loss_dict[k] = v.mean()

        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        return outputs["metrics"]

    def forward(self, ray_bundle, is_train=True) -> Dict[str, torch.Tensor]:
        if self.collider is not None:
            raise NotImplementedError
        return self.get_outputs(ray_bundle, is_train=is_train)
