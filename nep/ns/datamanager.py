from __future__ import annotations
import os
import torchvision as tv
from pathlib import Path
from tqdm import tqdm
import pickle
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Type, Union, Optional, Tuple, Dict
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.utils.rich_utils import CONSOLE
from nep.ns.buffer import GLOBALS
import random

import torch
import torch.nn.functional as F
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from jaxtyping import Int
from torch import Tensor, nn


class MyRayGenerator(RayGenerator):
    def forward(self, ray_indices: Int[Tensor, "num_rays 3"]) -> RayBundle:
        c = ray_indices[:, 0]  # camera indices
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices
        coords = self.image_coords[y, x]
        return self.cameras.generate_rays(camera_indices=c.unsqueeze(-1), coords=coords)


@dataclass
class NePDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: NePDataManager)
    save_cache: bool = True
    save_mask: bool = False


class NePDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """
    If mesh is not given:
        - sample pixels in next_train, next_eval, and return all pixels in next_eval_image
    else:
        - precompute all cpu rays and filter out bg rays (keep obj rays only) in next_train and next_eval
        - return all pixels (rays) in next_eval_image - to be processed in ns_model
    """

    config: NePDataManagerConfig

    def __init__(
        self,
        config: VanillaDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        trace_fn: Any = None,
        **kwargs,
    ):
        self.trace_fn: Callable = trace_fn
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)

    def trace_mesh_in_chunks(self, rays_o, rays_d, batch_size=1024**2, cpu=False):
        inters, normals, depth, hit_mask = [], [], [], []
        rn = rays_o.shape[0]
        for ri in range(0, rn, batch_size):
            inters_cur, normals_cur, depth_cur, hit_mask_cur = self.trace_fn(
                rays_o[ri : ri + batch_size], rays_d[ri : ri + batch_size]
            )
            if cpu:
                inters_cur = inters_cur.cpu()
                normals_cur = normals_cur.cpu()
                depth_cur = depth_cur.cpu()
                hit_mask_cur = hit_mask_cur.cpu()
            inters.append(inters_cur)
            normals.append(normals_cur)
            depth.append(depth_cur)
            hit_mask.append(hit_mask_cur)
        return (
            torch.cat(inters, 0),
            torch.cat(normals, 0),
            torch.cat(depth, 0),
            torch.cat(hit_mask, 0),
        )

    def get_all_obj_rays(self, split="train"):
        if split == "train":
            ray_generator = MyRayGenerator(
                self.train_dataset.cameras,
                self.train_camera_optimizer,
            )
            if not (
                self.config.train_num_times_to_repeat_images
                == self.config.train_num_images_to_sample_from
                == -1
            ):
                CONSOLE.print(
                    "[bold yellow]Warning: Intend to pre-computing all rays, but only a part of images are fetched."
                )
            all_image_batch = next(self.iter_train_image_dataloader)
            dataset = self.train_dataset

        elif split == "eval":
            ray_generator = MyRayGenerator(
                self.eval_dataset.cameras,
                self.eval_camera_optimizer,
            )
            all_image_batch = next(self.iter_eval_image_dataloader)
            dataset = self.eval_dataset

        else:
            raise NotImplementedError

        # get all rays
        assert isinstance(all_image_batch, dict)
        img_num = all_image_batch["image"].shape[0]
        if img_num < len(dataset):
            CONSOLE.print(
                f"[bold yellow]Warning: Only {img_num} / {len(dataset)} images are processed to get all cpu rays"
            )

        n, h, w, _ = all_image_batch["image"].shape
        pixel_sampler = self._get_pixel_sampler(dataset, n * h * w)
        batch = pixel_sampler.sample(all_image_batch)
        ray_indices = batch["indices"]

        # render mask only
        if self.config.save_mask:
            mask_folder = self.config.data / "masks"
            mask_folder.mkdir(exist_ok=True)
            print(f"Saving mask to {mask_folder}...")
            uv = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1).reshape(-1, 2)
            for im_idx in tqdm(range(n)):
                dataset = self.train_dataset if split == "train" else self.eval_dataset
                fname = dataset._dataparser_outputs.image_filenames[im_idx].name
                if (mask_folder / fname).exists():
                    continue
                im_idx_tensor = torch.full((h * w, 1), im_idx)
                chunk_ray_indices = torch.cat([im_idx_tensor, uv], -1)
                raybundle = ray_generator(chunk_ray_indices)
                inters, normals, depth, hit_mask = self.trace_mesh_in_chunks(
                    raybundle.origins.cuda(),
                    raybundle.directions.cuda(),
                    batch_size=h * w,
                    cpu=False,
                )
                hit_mask = hit_mask.cpu().reshape(h, w)
                mask = torch.zeros([h, w])
                # fg_uv = batch["indices"][batch["indices"][:, 0] == im_idx][:, 1:]
                # mask[fg_uv[:, 0], fg_uv[:, 1]] = 1
                mask[hit_mask] = 1
                tv.utils.save_image(mask, mask_folder / fname)

            print("Rendering mask done.")
            os._exit(0)

        # generate raybundle from ray_indices
        print(f"Pre-computing all cpu rays for {split} data based on mesh intersections...")
        all_hit_mask = []
        metadata = {"normals": [], "inters": [], "depth": []}
        data = {"origins": [], "directions": [], "pixel_area": [], "camera_indices": []}
        old_metadata = {"human_poses": [], "directions_norm": []}

        folder = GLOBALS["meshdir"] if "meshdir" in GLOBALS else Path("meshes")
        cachefile = folder / f"{split}_trace_results_{len(dataset)}.pt2"
        if cachefile.exists():
            try:
                print(f"Loading cached trace results from {cachefile}")
                # all_hit_mask, all_inter, all_normal, all_depth = torch.load(cachefile)
                with open(cachefile, "rb") as f:
                    # raybundle, batch = pickle.load(f)
                    all_hit_mask, metadata, data, old_metadata = pickle.load(f)
                # return raybundle, batch
            except Exception as e:
                print(e)
                print("Failed to load cached trace results. Re-computing...")

        # chunk forward
        else:
            for chunk_ray_indices in tqdm(torch.split(ray_indices, 1024 * 1024 * 2)):
                raybundle = ray_generator(chunk_ray_indices)
                inters, normals, depth, hit_mask = self.trace_mesh_in_chunks(
                    raybundle.origins.cuda(),
                    raybundle.directions.cuda(),
                    batch_size=1024 * 512,
                    cpu=False,
                )
                hit_mask = hit_mask.squeeze(-1).cpu()
                metadata["normals"].append(normals[hit_mask])
                metadata["inters"].append(inters[hit_mask])
                metadata["depth"].append(depth[hit_mask])
                all_hit_mask.append(hit_mask)

                for k in data:
                    data[k].append(getattr(raybundle, k)[hit_mask])

                for k in raybundle.metadata:
                    assert k in old_metadata
                    old_metadata[k].append(raybundle.metadata[k][hit_mask])

            old_metadata = {k: torch.cat(old_metadata[k]) for k in old_metadata.keys()}
            metadata = {k: torch.cat(v) for k, v in metadata.items()}

        raybundle = RayBundle(
            origins=torch.cat(data["origins"]),
            directions=torch.cat(data["directions"]),
            pixel_area=torch.cat(data["pixel_area"]),
            camera_indices=torch.cat(data["camera_indices"]),
            metadata={**old_metadata, **metadata},
        )

        hit_mask = torch.cat(all_hit_mask, 0)
        batch = {k: v[hit_mask] for k, v in batch.items()}

        # save cache
        if (
            self.config.save_cache
            and not self.dataparser.config.eval_only
            and not cachefile.exists()
        ):
            print(f"Saving trace results to {cachefile}")
            with open(cachefile, "wb") as f:
                # pickle.dump((raybundle, batch), f)
                pickle.dump((all_hit_mask, metadata, data, old_metadata), f)

        return raybundle, batch

    def setup_train(self):
        super().setup_train()
        if self.trace_fn is None:
            return

        # with mesh:
        self.train_obj_raybundle, self.all_train_ray_batch = self.get_all_obj_rays("train")
        # self.num_train_rays = len(self.train_obj_raybundle)

    def setup_eval(self):
        super().setup_eval()
        if self.trace_fn is None:
            return

        # with mesh:
        self.eval_obj_raybundle, self.all_eval_ray_batch = self.get_all_obj_rays("eval")

    # def _shuffle_train_batch(self):
    #     self.curr_train_ray_idx = 0
    #     shuffle_idxs = torch.randperm(self.num_train_rays, device='cpu')
    #     for k, v in self.train_batch.items():
    #         self.train_batch[k] = v[shuffle_idxs]

    def sample_raybundle(self, raybundle, batch, num):
        indices = torch.tensor(random.sample(range(len(raybundle)), k=num)).to(
            raybundle.origins.device
        )
        return raybundle[indices].to(self.device), {
            k: v[indices].to(self.device) for k, v in batch.items()
        }

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        if self.trace_fn is None:
            return super().next_train(step)

        self.train_count += 1
        return self.sample_raybundle(
            self.train_obj_raybundle,
            self.all_train_ray_batch,
            self.config.train_num_rays_per_batch,
        )

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        if self.trace_fn is None:
            return super().next_eval(step)

        self.eval_count += 1
        return self.sample_raybundle(
            self.eval_obj_raybundle,
            self.all_eval_ray_batch,
            self.config.eval_num_rays_per_batch,
        )

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        return super().next_eval_image(step)
