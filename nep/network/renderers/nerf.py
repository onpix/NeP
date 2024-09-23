import torch
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from typing import Tuple
from nep.utils.raw_utils import linear_to_srgb
import torch.nn as nn
import nerfacc
from torch import Tensor
from nerfstudio.model_components.ray_samplers import NeuSSampler
from nerfstudio.cameras.rays import Frustums, RayBundle
from nep.network.renderers.shape import NePShapeRenderer


class NeRFRenderer(NePShapeRenderer):
    def __init__(self, cfg, trace_fn=None):
        super().__init__(cfg)
        assert trace_fn is None  # stage 1 without mesh
        self.sdf_network = (
            self.deviation_network
        ) = self.color_network = self.sdf_inter_fun = nn.Identity()

    def ns_render(self, ray_bundle, perturb_overrite, anneal, is_train=True, step=None):
        if self.cfg.nerf.sampler == "grid" and is_train:
            self.update_estimators(step)

        ray_bundle.fars = torch.full(ray_bundle.shape, self.cfg.nerf.far).to(self.device)[..., None]
        ray_bundle.nears = torch.full_like(ray_bundle.fars, self.cfg.nerf.near)

        res = self.query_outer_nerf(
            ray_bundle, self.cfg.nerf.sampler, returns=["rgb", "depth"], step=step
        )
        res["metrics"] = {}
        res["ray_rgb"] = res.pop("rgb")
        return res
