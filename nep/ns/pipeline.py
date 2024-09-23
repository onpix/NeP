from typing import Literal, Type, Optional
import torch.distributed as dist
from nerfstudio.models.base_model import Model, ModelConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp.grad_scaler import GradScaler
import typing
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from nerfstudio.utils import profiler
from nerfstudio.pipelines.base_pipeline import (
    Pipeline,
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nep.network.renderers.material import NePMaterialRenderer

from .datamanager import NePDataManager
from nep.ns.models.nep import NePModel, NePModelConfig
import open3d
import numpy as np
from nep.utils.base_utils import check_memory_usage


@dataclass
class NePPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: NePPipeline)
    mesh: Optional[str] = None
    """object mesh path for raytracing and getting intersections."""


class NePPipeline(VanillaPipeline):
    def __init__(
        self,
        config: NePPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        Pipeline.__init__(self)

        # load mesh
        self.trace_fn = None
        if config.mesh is not None:
            from nep import raytracing

            print("loading mesh: ", config.mesh)
            self.mesh = open3d.io.read_triangle_mesh(config.mesh)
            self.ray_tracer = raytracing.RayTracer(
                np.asarray(self.mesh.vertices), np.asarray(self.mesh.triangles)
            )
            self.trace_fn = lambda o, d: self.get_ray_mesh_intersections(o, d)
            self.warned_normal = False

        # copied from father, but:
        #   - passing self.trace_fn to datamanager and model
        self.config = config
        self.test_mode = test_mode
        self.datamanager: NePDataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            trace_fn=self.trace_fn,
        )
        self.datamanager.to(device)
        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            trace_fn=self.trace_fn,
        )
        self.model.to(device)
        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])

    def predict_materials(self, batch_size=8192):
        verts = torch.from_numpy(np.asarray(self.mesh.vertices, np.float32)).cuda().float()
        metallic, roughness, albedo = [], [], []
        assert isinstance(self.model.network, NePMaterialRenderer)
        for vi in range(0, verts.shape[0], batch_size):
            m, r, a = self.model.network.shader_network.predict_materials(
                verts[vi : vi + batch_size]
            )
            # note: we assume predictions are squared roughness!!!
            r = torch.sqrt(torch.clamp(r, min=1e-7))
            metallic.append(m.detach().cpu().numpy())
            roughness.append(r.detach().cpu().numpy())
            albedo.append(a.detach().cpu().numpy())

        return {
            "metallic": np.concatenate(metallic, 0),
            "roughness": np.concatenate(roughness, 0),
            "albedo": np.concatenate(albedo, 0),
        }

    def get_ray_mesh_intersections(self, rays_o, rays_d):
        inters, normals, depth = self.ray_tracer.trace(rays_o, rays_d)
        depth = depth.reshape(*depth.shape, 1)
        normals = -normals
        normals = F.normalize(normals, dim=-1)
        if not self.warned_normal:
            print(
                "warn!!! the normals are flipped in NeuS by default. You may flip the normal according to your mesh!"
            )
            self.warned_normal = True
        miss_mask = depth >= 10
        hit_mask = ~miss_mask
        return inters, normals, depth, hit_mask

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        return super().get_train_loss_dict(step)

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle, is_train=False)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        self.eval()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict
