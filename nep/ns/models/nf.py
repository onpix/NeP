import torch
import numpy as np
import torch.nn.functional as F
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
import torchvision as tv
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type
from nerfstudio.models.nerfacto import NerfactoField, NerfactoModelConfig, NerfactoModel
from nero.utils.draw_utils import add_text_to_image


@dataclass
class MyNerfactoModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: MyNerfactoModel)
    get_inner: bool = False
    """predict inner rgb"""


class MyNerfactoModel(NerfactoModel):
    config: MyNerfactoModelConfig

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        if self.config.get_inner:
            images_dict.pop("img")
        else:
            return {}, {
                "img": images_dict["img"],
                "depth": images_dict["depth"] / images_dict["depth"].max(),
            }

        images_dict["gt"] = batch["image"].to(self.device)[..., :3]
        for k, v in outputs.items():
            if k.startswith("inner"):
                images_dict[k] = add_text_to_image(v.detach().cpu(), k, color="red")

        print("Save names:", images_dict.keys())
        combined_rgb = torch.stack([x.detach().cpu() for x in images_dict.values()]).permute(
            0, 3, 1, 2
        )
        combined_rgb = tv.utils.make_grid(combined_rgb, nrow=5).permute(1, 2, 0)
        return metrics_dict, {"vis": combined_rgb}

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

        # get inner rgb
        outputs = {}
        if self.config.get_inner:
            for factor in np.linspace(1.2, 5, 20):
                mask = torch.norm(ray_samples.frustums.get_positions(), dim=-1) < 1 / factor
                inner_rgbs = field_outputs[FieldHeadNames.RGB].clone()
                inner_weights = weights.clone()
                inner_rgbs[~mask] = 0
                inner_weights[~mask] = 0
                inner_rgb = self.renderer_rgb(rgb=inner_rgbs, weights=inner_weights)
                outputs[f"inner_{round(factor, 2)}"] = inner_rgb

        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs.update(
            {
                "rgb": rgb,
                "accumulation": accumulation,
                "depth": depth,
                "expected_depth": expected_depth,
            }
        )
        if self.config.predict_normals:
            normals = self.renderer_normals(
                normals=field_outputs[FieldHeadNames.NORMALS], weights=weights
            )
            pred_normals = self.renderer_normals(
                field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights
            )
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i]
            )

        return outputs