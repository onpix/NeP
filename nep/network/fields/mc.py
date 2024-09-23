from typing import Optional, Mapping
import torchvision as tv
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch import Tensor
from rich import print
from omegaconf import DictConfig
import numpy as np

from nep.utils.base_utils import (
    az_el_to_points,
    sample_sphere,
    roughness2area,
)
from nep.utils.raw_utils import linear_to_srgb
from nep.utils.ref_utils import generate_ide_fn, calc_ref_dir
from nerfstudio.cameras.rays import RayBundle
from nep.network.fields.utils import (
    make_predictor,
    get_embedder,
    get_camera_plane_intersection,
    get_sphere_intersection,
    offset_points_to_sphere,
    IPE,
)
from nep.network.renderers.shape import NePShapeRenderer
from nep.network.renderers.nerf import NeRFRenderer


class MaterialFeatsNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_enc, input_dim = get_embedder(8, 3)
        run_dim = 256
        weight_norm = torch.nn.utils.weight_norm
        self.module0 = nn.Sequential(
            weight_norm(nn.Linear(input_dim, run_dim)),
            nn.ReLU(),
            weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
        )
        self.module1 = nn.Sequential(
            weight_norm(nn.Linear(input_dim + run_dim, run_dim)),
            nn.ReLU(),
            weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            weight_norm(nn.Linear(run_dim, run_dim)),
        )

    def forward(self, x):
        x = self.pos_enc(x)
        input = x
        x = self.module0(x)
        return self.module1(torch.cat([x, input], -1))


def saturate_dot(v0, v1):
    return torch.clamp(torch.sum(v0 * v1, dim=-1, keepdim=True), min=0.0, max=1.0)


class MCShadingNetwork(nn.Module):
    default_cfg = {}

    def __init__(self, cfg, ray_trace_fun, bg_model: Optional[NePShapeRenderer] = None):
        # self.cfg = {**self.default_cfg, **cfg}
        self.cfg: DictConfig = cfg
        super().__init__()
        print("MCShadingNetwork init:", self.cfg)

        # light part
        self.sph_enc = generate_ide_fn(4)
        ide_dim = 38
        # self.sph_enc = generate_ide_fn(5)
        # self.dir_enc, dir_dim = get_embedder(6, 3)
        self.pos_enc, pos_dim = get_embedder(8, 3)
        self.bg_model: NePShapeRenderer = None
        make_outer_light = lambda inc: make_predictor(
            inc, 3, activation=self.cfg.outer_light_act, exp_max=self.cfg.light_exp_max
        )
        if "direction" in self.cfg.outer_light_version:
            self.outer_light = make_outer_light(ide_dim)
        elif "sphere_direction" in self.cfg.outer_light_version:
            self.outer_light = make_outer_light(ide_dim * 2)
        elif "5d" in self.cfg.outer_light_version:
            self.outer_light = make_outer_light(ide_dim + pos_dim)

        # check: can only use one of them
        cond = sum(
            [
                "direction" in self.cfg.outer_light_version,
                "sphere_direction" in self.cfg.outer_light_version,
                "5d" in self.cfg.outer_light_version,
            ]
        )
        assert cond in [0, 1]

        if (
            "nerf" in self.cfg.outer_light_version
            or self.cfg.plen_light is not None
            or self.cfg.inner_light == "neus"
            or self.cfg.bake_loss > 0
        ):
            self.bg_model = bg_model
            assert self.bg_model is not None
            if "roughness_transform" in self.cfg:
                self.roughness_transform = eval(self.cfg.roughness_transform)

        # material part
        if self.cfg.with_stage1_mlps:
            assert self.bg_model is not None
            self.feats_network = lambda x: self.bg_model.sdf_network(x, with_grad=False)[:, 1:]
            self.metallic_predictor = self.bg_model.color_network.metallic_predictor
            self.roughness_predictor = self.bg_model.color_network.roughness_predictor
            self.albedo_predictor = self.bg_model.color_network.albedo_predictor
        else:
            self.feats_network = MaterialFeatsNetwork()
            self.metallic_predictor = make_predictor(256 + 3, 1)
            self.roughness_predictor = make_predictor(256 + 3, 1)
            self.albedo_predictor = make_predictor(256 + 3, 3)

        # lights
        if hasattr(self, "outer_light"):
            nn.init.constant_(self.outer_light[-2].bias, np.log(0.5))

        if self.cfg.human_light:
            self.human_light = make_predictor(2 * 2 * 6, 4, activation=self.cfg.mc_light_act)
            nn.init.constant_(self.human_light[-2].bias, np.log(0.02))

        self.inner_light = make_predictor(
            pos_dim + ide_dim,
            3,
            activation=self.cfg.mc_light_act,
            exp_max=self.cfg.inner_light_exp_max,
        )
        self.inner_diffuse_light = make_predictor(
            pos_dim,
            3,
            activation=self.cfg.mc_light_act,
            exp_max=self.cfg.inner_light_exp_max,
        )
        nn.init.constant_(self.inner_light[-2].bias, np.log(0.5))

        # predefined diffuse sample directions
        az, el = sample_sphere(self.cfg.diffuse_sample_num, 0)
        az, el = az * 0.5 / np.pi, 1 - 2 * el / np.pi  # scale to [0,1]
        self.diffuse_direction_samples = np.stack([az, el], -1)
        self.diffuse_direction_samples = torch.from_numpy(
            self.diffuse_direction_samples.astype(np.float32)
        ).cuda()  # [dn0,2]

        az, el = sample_sphere(self.cfg.specular_sample_num, 0)
        az, el = az * 0.5 / np.pi, 1 - 2 * el / np.pi  # scale to [0,1]
        self.specular_direction_samples = np.stack([az, el], -1)
        self.specular_direction_samples = torch.from_numpy(
            self.specular_direction_samples.astype(np.float32)
        ).cuda()  # [dn1,2]

        az, el = sample_sphere(8192, 0)
        light_pts = az_el_to_points(az, el)
        self.register_buffer("light_pts", torch.from_numpy(light_pts.astype(np.float32)))
        self.ray_trace_fun = ray_trace_fun

    def get_orthogonal_directions(self, directions):
        x, y, z = torch.split(directions, 1, dim=-1)  # pn,1
        otho0 = torch.cat([y, -x, torch.zeros_like(x)], -1)
        otho1 = torch.cat([-z, torch.zeros_like(x), x], -1)
        mask0 = torch.norm(otho0, dim=-1) > torch.norm(otho1, dim=-1)
        mask1 = ~mask0
        otho = torch.zeros_like(directions)
        otho[mask0] = otho0[mask0]
        otho[mask1] = otho1[mask1]
        otho = F.normalize(otho, dim=-1)
        return otho

    def sample_diffuse_directions(self, normals, is_train):
        # normals [pn,3]
        z = normals  # pn,3
        x = self.get_orthogonal_directions(normals)  # pn,3
        y = torch.cross(z, x, dim=-1)  # pn,3
        # y = torch.cross(z, x, dim=-1) # pn,3

        # project onto this tangent space
        az, el = torch.split(self.diffuse_direction_samples, 1, dim=1)  # sn,1
        el, az = el.unsqueeze(0), az.unsqueeze(0)
        az = az * np.pi * 2
        el_sqrt = torch.sqrt(el + 1e-7)
        if is_train and self.cfg.random_azimuth:
            az = (az + torch.rand(z.shape[0], 1, 1) * np.pi * 2) % (2 * np.pi)
        coeff_z = torch.sqrt(1 - el + 1e-7)
        coeff_x = el_sqrt * torch.cos(az)
        coeff_y = el_sqrt * torch.sin(az)

        directions = (
            coeff_x * x.unsqueeze(1) + coeff_y * y.unsqueeze(1) + coeff_z * z.unsqueeze(1)
        )  # pn,sn,3
        return directions

    def sample_specular_directions(self, reflections, roughness, is_train):
        # roughness [pn,1]
        z = reflections  # pn,3
        x = self.get_orthogonal_directions(reflections)  # pn,3
        y = torch.cross(z, x, dim=-1)  # pn,3
        roughness = self.get_squared_roughness(roughness)
        a = roughness  # we assume the predicted roughness is already squared

        az, el = torch.split(self.specular_direction_samples, 1, dim=1)  # sn,1
        phi = np.pi * 2 * az  # sn,1     # phi is actually the original az?
        a, el = a.unsqueeze(1), el.unsqueeze(0)  # [pn,1,1] [1,sn,1]
        cos_theta = torch.sqrt(
            (1.0 - el + 1e-6) / (1.0 + (a**2 - 1.0) * el + 1e-6) + 1e-6
        )  # pn,sn,1
        sin_theta = torch.sqrt(1 - cos_theta**2 + 1e-6)  # pn,sn,1

        phi = phi.unsqueeze(0)  # 1,sn,1
        if is_train and self.cfg.random_azimuth:
            phi = (phi + torch.rand(z.shape[0], 1, 1) * np.pi * 2) % (2 * np.pi)
        coeff_x = torch.cos(phi) * sin_theta  # pn,sn,1
        coeff_y = torch.sin(phi) * sin_theta  # pn,sn,1
        coeff_z = cos_theta  # pn,sn,1

        # convert from local coordinate -> world coordinate
        directions = (
            coeff_x * x.unsqueeze(1) + coeff_y * y.unsqueeze(1) + coeff_z * z.unsqueeze(1)
        )  # pn,sn,3
        return directions

    def get_inner_lights(
        self, origins, inters, neg_dirs, pixel_areas, normals, depth, type="mlp", level=0
    ):
        """
        predict color given the reflections dir and points.
        predict the incident light color!
        note that neg_dirs here is emitted from points to the viewer (origins).
        todo: for thoes points, only the reflection color is related to the reflection direction. what if those points are from the diffuse color query?
        """
        if type == "nerf":
            # query nerf to get the rgb as lighting color
            # for those points, interval length is fixed to 0.001
            # one sample for each ray.
            assert pixel_areas is not None
            neg_dirs = F.normalize(neg_dirs, dim=-1)
            interval = 0.001
            ray_bundle = RayBundle(origins=origins, directions=-neg_dirs, pixel_area=pixel_areas)
            ray_samples = ray_bundle.get_ray_samples(
                bin_starts=depth[..., None] - interval / 2, bin_ends=depth[..., None] + interval / 2
            )
            gaussian_samples = ray_samples.frustums.get_gaussian_blob()
            dists = torch.zeros_like(depth) + interval
            _, color, _ = self.bg_model.outer_nerf_forward_samples(
                inters,
                dists,
                neg_dirs,
                cov=gaussian_samples.cov[:, 0, ...],
                normals=F.normalize(normals, dim=-1),  # use neus normals
            )
            return color

        # nep version
        elif type == "mlp":
            pos_enc = self.pos_enc(inters)
            normals = F.normalize(normals, dim=-1)
            neg_dirs = F.normalize(neg_dirs, dim=-1)
            reflections = calc_ref_dir(neg_dirs, normals)
            dir_enc = self.sph_enc(reflections, 0)
            return self.inner_light(torch.cat([pos_enc, dir_enc], -1))

        elif type == "neus":
            # neus points
            assert not isinstance(self.bg_model, NeRFRenderer)

            # if self.cfg.freeze_neus:
            # self.bg_model.sdf_network.eval()
            # self.bg_model.color_network.eval()

            with torch.set_grad_enabled(not self.cfg.freeze_neus):
                feature, normals = self.bg_model.sdf_network(inters, with_grad=True)
                # sdf = feature[..., 0]
                feature = feature[..., 1:]
                self.bg_model.color_network.cfg["human_light"] = False
                inner_rgb, occ_info = self.bg_model.color_network(
                    inters, normals, neg_dirs, feature, None, step=0
                )
                return inner_rgb

        elif type == "neus_ray":
            # neus ray marching
            assert not isinstance(self.bg_model, NeRFRenderer)
            ...

        elif type in ["trace", "trace_detach"]:
            metallic, roughness, albedo = self.predict_materials(inters)
            normals = F.normalize(normals, dim=-1)
            neg_dirs = F.normalize(neg_dirs, dim=-1)
            ref_dirs = calc_ref_dir(neg_dirs, normals)
            inters_rgb = self.shade_mixed(
                inters,
                normals,
                neg_dirs,
                ref_dirs,
                metallic,
                roughness,
                albedo,
                None,
                self.training,
                level=level + 1,
            )
            if type == "trace_detach":
                inters_rgb = inters_rgb.detach()
            return inters_rgb

        else:
            raise NotImplementedError

    def get_human_light(self, points, directions, human_poses):
        inter, dists, hits = get_camera_plane_intersection(points, directions, human_poses)
        scale_factor = 0.3
        mean = inter[..., :2] * scale_factor
        hits = hits & (torch.norm(mean, dim=-1) < 1.5) & (dists > 0)
        hits = hits.float().unsqueeze(-1)
        mean = mean * hits

        var = torch.zeros_like(mean)
        pos_enc = IPE(mean, var, 0, 6)  # 2*2*6
        human_lights = self.human_light(pos_enc)
        human_lights = human_lights * hits
        human_lights, human_weights = human_lights[..., :3], human_lights[..., 3:]
        human_weights = torch.clamp(human_weights, max=1.0, min=0.0)
        return human_lights, human_weights

    def get_outer_lights(self, origins, ray_dirs, level, pixel_area=None):
        # here `directions` are emitted from points to background.
        light_types = self.cfg.outer_light_version
        lights = []
        if "direction" in light_types:
            outer_enc = self.sph_enc(ray_dirs, 0)
            outer_lights = self.outer_light(outer_enc) * self.cfg.outer_mlp_scale
            lights.append(outer_lights)

        elif "sphere_direction" in light_types:
            outer_dirs = ray_dirs
            outer_pts = origins
            outer_enc = self.sph_enc(outer_dirs, 0)

            # for those points at the border of the scene.
            mask = torch.norm(outer_pts, dim=-1) > self.cfg.outer_shrink
            if torch.sum(mask) > 0:
                outer_pts = torch.clone(outer_pts)
                outer_pts[
                    mask
                ] *= self.cfg.outer_shrink  # shrink this point a little bit to get the intersection

            dists = get_sphere_intersection(outer_pts, outer_dirs)
            sphere_pts = outer_pts + outer_dirs * dists
            sphere_pts = self.sph_enc(sphere_pts, 0)
            outer_lights = (
                self.outer_light(torch.cat([outer_enc, sphere_pts], -1)) * self.cfg.outer_mlp_scale
            )
            lights.append(outer_lights)

        elif "5d" in light_types:
            outer_dirs = ray_dirs
            outer_pts = origins
            outer_enc = self.sph_enc(outer_dirs, 0)

            origins_enc = self.pos_enc(outer_pts)

            # for those points at the border of the scene.
            mask = torch.norm(outer_pts, dim=-1) > self.cfg.outer_shrink
            if torch.sum(mask) > 0:
                outer_pts = torch.clone(outer_pts)
                outer_pts[
                    mask
                ] *= self.cfg.outer_shrink  # shrink this point a little bit to get the intersection

            # dists = get_sphere_intersection(outer_pts, outer_dirs)
            # sphere_pts = outer_pts + outer_dirs * dists
            # sphere_pts = self.sph_enc(sphere_pts, 0)
            outer_lights = (
                self.outer_light(torch.cat([outer_enc, origins_enc], -1)) * self.cfg.outer_mlp_scale
            )
            lights.append(outer_lights)

        if "nerf" in light_types or (
            self.cfg.bake_loss > 0 and level == 0 and self.step % self.cfg.bake_every == 0
        ):
            # self.bg_model should be NeroShapeRenderer or NeRFRenderer
            assert self.bg_model is not None
            sampler = self.cfg.nerf_sampler
            rays_o, rays_d = origins, ray_dirs
            assert sampler in ["uniform", "pdf", "nep_bg"] and pixel_area is not None
            near, far = self.bg_model.near_far_from_sphere(rays_o, rays_d)
            nerf_lights = self.bg_model.query_outer_nerf(
                RayBundle(
                    origins=rays_o,
                    directions=rays_d,
                    pixel_area=pixel_area,
                    nears=far,
                    fars=torch.full_like(far, 1e3),
                ),
                sampler=sampler,
                return_linear=self.cfg.nerf_use_linear,
            )["rgb"]
            lights.append(nerf_lights)

        # combine
        return lights

    def get_pixel_area(self, real_roughness, points_shape):
        pixel_area = None
        if self.bg_model and self.bg_model.use_mip:
            if self.cfg.detach_roughness:
                real_roughness = real_roughness.detach()

            # create cone for each ray
            pixel_area = roughness2area(
                real_roughness,
                max_tan=self.cfg.max_tan,
                roughness_transform=self.roughness_transform,
            )
            pixel_area = pixel_area.unsqueeze(-1).expand(*points_shape, 1)
        return pixel_area

    def get_plenoptic_lights(self, points, directions, human_poses, roughness, eps):
        # todo: we may use occ-grid to help sampling here, bacause sampling withn sphere and sampling uniformly in the background are not good.
        shape = points.shape[:-1]  # pn,sn
        biased_points = points + directions * eps
        roughness = self.get_real_roughness(roughness)
        pixel_areas = self.get_pixel_area(roughness, shape)
        assert pixel_areas is not None

        flat_points = biased_points.view(-1, 3)
        flat_directions = directions.view(-1, 3)
        flat_pixel_areas = pixel_areas.reshape(-1, 1)
        raybundle = RayBundle(
            origins=flat_points,
            directions=flat_directions,
            pixel_area=flat_pixel_areas,
        )

        if self.cfg.plen_light == "nerf":
            raise NotImplementedError
            raybundle.nears, raybundle.fars = 0.01, 1e3
            plen_lights = self.bg_model.query_outer_nerf(
                raybundle, sampler=self.cfg.nerf_sampler, return_linear=self.cfg.nerf_use_linear
            )["rgb"]

        elif self.cfg.plen_light == "nep":
            raybundle.nears, raybundle.fars = self.bg_model.near_far_from_sphere(
                raybundle.origins, raybundle.directions
            )
            raybundle.metadata["human_poses"] = human_poses.view(-1, 12)
            with torch.no_grad():
                # freeze nep1 model
                self.bg_model.eval()
                plen_lights = self.bg_model.ns_render(raybundle, 0, 0, False, 0, False)["ray_rgb"]

        # here human_light could be predicted with both direct and indirect lights.
        # todo: may need to check the correctness of this part, as the indirect light should not be considered.
        if self.cfg.human_light and points.shape[0]:
            human_lights, human_weights = self.get_human_light(
                points.view(-1, 3),
                flat_directions,
                human_poses.view(-1, 3, 4),
            )
        else:
            human_lights, human_weights = torch.zeros_like(plen_lights), torch.zeros(
                plen_lights.shape[0], 1
            )
        human_lights = human_lights.view(*shape, 3)
        human_weights = human_weights.view(*shape, 1)
        plen_lights = plen_lights.view(*shape, 3)
        lights = plen_lights * (1 - human_weights) + human_lights * human_weights
        # near_mask = (depth > eps).float()
        # lights = lights * near_mask  # very near surface does not bring lights
        return lights, human_lights * human_weights

    def get_lights(self, origins, ray_dirs, human_poses, roughness, level=0):
        shape = origins.shape[:-1]  # pn,sn
        eps = 1e-5  # a small that is used to avoid self-intersection

        # forward mipnerf 360 to get all lights (both bg and fg)
        if self.cfg.plen_light is not None:
            return self.get_plenoptic_lights(origins, ray_dirs, human_poses, roughness, eps)

        # why: here we trace the rays emitted from the mesh surface and see if they have intersections with mesh itself.
        inters, normals, depth, hit_mask = self.ray_trace_fun(
            origins.reshape(-1, 3) + ray_dirs.reshape(-1, 3) * eps,
            ray_dirs.reshape(-1, 3),
        )
        inters, normals, depth, hit_mask = (
            inters.reshape(*shape, 3),
            normals.reshape(*shape, 3),
            depth.reshape(*shape, 1),
            hit_mask.reshape(*shape),
        )
        miss_mask = ~hit_mask

        # hit_mask
        lights = torch.zeros(*shape, 3)
        human_lights, human_weights = torch.zeros([1, 3]), torch.zeros([1, 1])
        if level == 0:
            vis_results = {
                f"outer_light{i}": torch.zeros_like(lights)
                for i in range(len(self.cfg.outer_light_version))
            }
            if self.cfg.outer_light_reduce.startswith("sky"):
                vis_results["sky"] = torch.zeros(*lights.shape[:-1], 1)
            ex_loss = {}
        else:
            vis_results = None
            ex_loss = None

        # for those hitting env map
        roughness = self.get_real_roughness(roughness)
        pixel_areas = self.get_pixel_area(roughness, shape)
        if torch.sum(miss_mask) > 0:
            # diffuse_miss_mask = miss_mask[:, :self.cfg['diffuse_sample_num']]
            # specular_miss_mask = miss_mask[:, self.cfg['diffuse_sample_num']:]

            outer_lights_list = self.get_outer_lights(
                origins[miss_mask],
                ray_dirs[miss_mask],
                level,
                pixel_areas[miss_mask] if pixel_areas is not None else None,
            )
            if self.cfg.bake_loss > 0:
                self.cfg.outer_light_reduce = "bake"

            if self.cfg.outer_light_reduce == "mean":
                outer_lights = sum(outer_lights_list) / len(outer_lights_list)
            elif self.cfg.outer_light_reduce == "sum":
                outer_lights = sum(outer_lights_list)
            elif self.cfg.outer_light_reduce == "sum01":
                outer_lights = sum(outer_lights_list).clip(0, 1)

            elif self.cfg.outer_light_reduce == "bake":
                assert self.cfg.outer_light_version == [
                    "sphere_direction"
                ] or self.cfg.outer_light_version == ["5d"]
                if len(outer_lights_list) == 2:
                    # weight = float(self.cfg.outer_light_reduce.replace("bake", ""))
                    mlp_lights, nerf_lights = outer_lights_list
                    ex_loss["loss_bakeL"] = (
                        torch.mean((mlp_lights - nerf_lights.detach()) ** 2) * self.cfg.bake_loss
                    )
                    outer_lights = mlp_lights
                else:
                    outer_lights = outer_lights_list[0]

            elif self.cfg.outer_light_reduce == "sky":
                sky_weights = (1 + ray_dirs[miss_mask][:, -1]) / 2
                mlp_lights, nerf_lights = outer_lights_list
                outer_lights = mlp_lights * sky_weights[:, None] + nerf_lights * (
                    1 - sky_weights[:, None]
                )
                outer_lights *= self.cfg.outer_light_scale
                if level == 0:
                    vis_results["sky"][miss_mask] = sky_weights[:, None]

            elif self.cfg.outer_light_reduce == "sky2":
                sky_weights = 1 + ray_dirs[miss_mask][:, -1]
                mlp_lights, nerf_lights = outer_lights_list
                outer_lights = mlp_lights * sky_weights[:, None] + nerf_lights
                outer_lights *= self.cfg.outer_light_scale
                if level == 0:
                    vis_results["sky"][miss_mask] = sky_weights[:, None]

            elif self.cfg.outer_light_reduce == "sky3":
                sky_weights = (1 + ray_dirs[miss_mask][:, -1]) / 2
                mlp_lights, nerf_lights = outer_lights_list
                outer_lights = mlp_lights + nerf_lights * (1 - sky_weights[:, None])
                outer_lights *= self.cfg.outer_light_scale
                if level == 0:
                    vis_results["sky"][miss_mask] = sky_weights[:, None]
            else:
                raise NotImplementedError

            if self.cfg.clip_light is not None:
                outer_lights = torch.clamp(outer_lights, max=self.cfg.clip_light)

            if self.cfg.human_light and level == 0:
                human_lights, human_weights = self.get_human_light(
                    origins[miss_mask], ray_dirs[miss_mask], human_poses[miss_mask]
                )
            else:
                human_lights, human_weights = torch.zeros_like(outer_lights), torch.zeros(
                    outer_lights.shape[0], 1
                )
            lights[miss_mask] = outer_lights * (1 - human_weights) + human_lights * human_weights

            # collect out lights for vis
            if level == 0:
                for i in range(len(self.cfg.outer_light_version)):
                    vis_results[f"outer_light{i}"][miss_mask] = outer_lights_list[i]

        # for those still hitting mesh, predict the incident color of these inters.
        if torch.sum(hit_mask) > 0:
            lights[hit_mask] = self.get_inner_lights(
                origins[hit_mask],
                inters[hit_mask],
                -ray_dirs[hit_mask],
                pixel_areas[hit_mask] if pixel_areas is not None else None,
                normals[hit_mask],
                depth[hit_mask],
                type=self.cfg.inner_light if level == 0 else "mlp",
                level=level,
            )

        near_mask = (depth > eps).float()
        lights = lights * near_mask  # very near surface does not bring lights
        return (
            lights,
            human_lights * human_weights,
            vis_results,
            ex_loss,
        )  # , inters, normals, hit_mask

    def fresnel_schlick(self, F0, HoV):
        return F0 + (1.0 - F0) * torch.clamp(1.0 - HoV, min=0.0, max=1.0) ** 5.0

    def fresnel_schlick_directions(self, F0, view_dirs, directions):
        H = view_dirs + directions  # [pn,sn,3]
        H = F.normalize(H, dim=-1)
        HoV = torch.clamp(
            torch.sum(H * view_dirs, dim=-1, keepdim=True), min=0.0, max=1.0
        )  # [pn,sn,1]
        fresnel = self.fresnel_schlick(F0, HoV)  # [pn,sn,1]
        return fresnel, H, HoV

    def get_real_roughness(self, roughness):
        if self.cfg.squared_roughness_fn == "identity":
            return roughness.sqrt()
        elif self.cfg.squared_roughness_fn == "invert":
            return 1 - roughness.sqrt()
        elif self.cfg.squared_roughness_fn == "invert2":
            return 1 - roughness
        elif self.cfg.squared_roughness_fn == "square":
            return roughness
        else:
            raise NotImplementedError

    def get_squared_roughness(self, roughness):
        if self.cfg.squared_roughness_fn == "identity":
            return roughness
        elif self.cfg.squared_roughness_fn == "invert":
            return 1 - roughness
        elif self.cfg.squared_roughness_fn == "invert2":
            return (1 - roughness) ** 2
        elif self.cfg.squared_roughness_fn == "square":
            return roughness**2
        else:
            raise NotImplementedError

    def geometry_schlick_ggx(self, NoV, roughness):
        # a = roughness  # a = roughness**2: we assume the predicted roughness is already squared
        roughness = self.get_squared_roughness(roughness)
        if self.cfg.correct_schlick:
            k = (roughness + 1) ** 2 / 8
        else:
            k = roughness / 2

        num = NoV
        denom = NoV * (1 - k) + k
        return num / (denom + 1e-5)

    def geometry_schlick(self, NoV, NoL, roughness):
        ggx2 = self.geometry_schlick_ggx(NoV, roughness)
        ggx1 = self.geometry_schlick_ggx(NoL, roughness)
        return ggx2 * ggx1

    def geometry_ggx_smith_correlated(self, NoV, NoL, roughness):
        def fun(alpha2, cos_theta):
            # cos_theta = torch.clamp(cos_theta,min=1e-7,max=1-1e-7)
            cos_theta2 = cos_theta**2
            tan_theta2 = (1 - cos_theta2) / (cos_theta2 + 1e-7)
            return 0.5 * torch.sqrt(1 + alpha2 * tan_theta2) - 0.5

        # todo: is this roughness squared?
        alpha_sq = roughness**2
        return 1.0 / (1.0 + fun(alpha_sq, NoV) + fun(alpha_sq, NoL))

    def predict_materials(self, pts):
        feats = self.feats_network(pts)
        metallic = self.metallic_predictor(torch.cat([feats, pts], -1))
        roughness = self.roughness_predictor(torch.cat([feats, pts], -1))
        rmax, rmin = 1.0, 0.04**2
        roughness = roughness * (rmax - rmin) + rmin
        # roughness = torch.clamp(roughness, min=rmin, max=rmax)
        albedo = self.albedo_predictor(torch.cat([feats, pts], -1))
        # if roughness.shape[0] > 0:
        #     print(float(roughness.max()), float(roughness.min()), file=open("roughness-ours.txt", "a"))
        return metallic, roughness, albedo

    def distribution_ggx(self, NoH, roughness):
        """
        the ggx distributed D function of cook-torrance brdf.
        """
        roughness = self.get_squared_roughness(roughness)
        a = roughness
        a2 = a**2
        NoH2 = NoH**2
        denom = NoH2 * (a2 - 1.0) + 1.0
        return a2 / (np.pi * denom**2 + 1e-4)

    def geometry(self, NoV, NoL, roughness):
        if self.cfg.geometry_type == "schlick":
            geometry = self.geometry_schlick(NoV, NoL, roughness)
        elif self.cfg.geometry_type == "ggx_smith":
            geometry = self.geometry_ggx_smith_correlated(NoV, NoL, roughness)
        else:
            raise NotImplementedError
        return geometry

    def shade_mixed(
        self,
        pts,
        normals,
        neg_dirs,
        ref_dirs,
        metallic,
        roughness,
        albedo,
        human_poses,
        is_train,
        level=0,
    ):
        """
        combine cook-torrance brdf (to get sepcular color) and lambertain brdf (to get diffuse color)
        cook-torrance: brdf = DGF / [4 (n · v) (n · l)]
        lambertain: k_d * albedo / pi

        neg_dirs: directions from points to viewer
        ref_dirs: directions that symmetric with the view_dirs
        level: light bounce time
        """
        F0 = 0.04 * (1 - metallic) + metallic * albedo  # [pn,1]

        # Step 1. sample diffuse directions uniformly
        diffuse_directions = self.sample_diffuse_directions(normals, is_train)  # [pn,sn0,3]
        point_num, diffuse_num, _ = diffuse_directions.shape

        # Step 2. sample specular directions in lobe only
        if self.cfg.single_specular_light:
            specular_directions = ref_dirs.unsqueeze(1)  # [pn,1,3]
        else:
            specular_directions = self.sample_specular_directions(
                ref_dirs, roughness, is_train
            )  # [pn,sn1,3]
        specular_num = specular_directions.shape[1]

        # debug: vis sampled directions
        # from utils.draw_utils import vis_directions
        # vis_directions([diffuse_directions[0], specular_directions[0], reflections[0:1]], './dirs.html')

        # diffuse sample prob
        NoL_d = saturate_dot(diffuse_directions, normals.unsqueeze(1))  # [pn, sn0, 1]
        diffuse_probability = NoL_d / np.pi * (diffuse_num / (specular_num + diffuse_num))

        # specualr sample prob. H_s is halfway-vecter
        H_s = neg_dirs.unsqueeze(1) + specular_directions  # [pn,sn0,3]
        H_s = F.normalize(H_s, dim=-1)
        NoH_s = saturate_dot(normals.unsqueeze(1), H_s)
        VoH_s = saturate_dot(neg_dirs.unsqueeze(1), H_s)
        specular_probability = (
            self.distribution_ggx(NoH_s, roughness.unsqueeze(1))
            * NoH_s
            / (4 * VoH_s + 1e-5)
            * (specular_num / (specular_num + diffuse_num))
        )  # D * NoH / (4 * VoH)

        # combine
        directions = torch.cat([diffuse_directions, specular_directions], 1)
        probability = torch.cat([diffuse_probability, specular_probability], 1)
        sn = diffuse_num + specular_num
        fresnel, H, HoV = self.fresnel_schlick_directions(
            F0.unsqueeze(1), neg_dirs.unsqueeze(1), directions
        )
        NoV = saturate_dot(normals, neg_dirs).unsqueeze(1)  # pn,1,3
        NoL = saturate_dot(normals.unsqueeze(1), directions)  # pn,sn,3
        geometry = self.geometry(NoV, NoL, roughness.unsqueeze(1))
        NoH = saturate_dot(normals.unsqueeze(1), H)
        distribution = self.distribution_ggx(NoH, roughness.unsqueeze(1))
        human_poses = (
            human_poses.unsqueeze(1).repeat(1, sn, 1, 1) if human_poses is not None else None
        )
        pts_ = pts.unsqueeze(1).repeat(1, sn, 1)
        lights, hl, outer_vis, ex_loss = self.get_lights(
            pts_, directions, human_poses, roughness, level
        )  # pn,sn,3

        # todo: save grid images for all lights rgb here.

        # todo: why here specular color is from both specular and diffuse results?
        # brdf value without F term for each light
        # here weights sum value is in 10 ~ 24; prob sum value is around 1.
        specular_weights = distribution * geometry / (4 * NoV * probability + 1e-5)

        specular_lights_real = lights[:, -specular_num:]
        if self.cfg.less_spec:
            specular_weights = specular_weights[:, -specular_num:]
            specular_lights = specular_lights_real * specular_weights

            # todo: fix here.
            # fresnel = fresnel[:, -specular_num:]
        else:
            specular_lights = lights * specular_weights

        specular_colors = torch.mean(fresnel * specular_lights, 1)  # [pn, 3]
        # specular_weights = specular_weights * fresnel

        # diffuse only consider diffuse directions
        kd = 1 - metallic.unsqueeze(1)
        diffuse_lights = lights[:, :diffuse_num]
        diffuse_colors = albedo.unsqueeze(1) * kd[:, :diffuse_num] * diffuse_lights
        diffuse_colors = torch.mean(diffuse_colors, 1)

        colors = diffuse_colors + specular_colors
        colors = linear_to_srgb(colors)
        if level > 0:
            return colors

        self.outputs: Mapping = {"metrics": {}}
        outputs = self.outputs
        outputs.update(ex_loss)
        outputs["albedo"] = albedo
        # outputs["all_lights"] = lights
        outputs["roughness"] = roughness
        outputs["metallic"] = metallic
        outputs["human_light"] = hl.reshape(-1, 3)

        # for reg loss - calculate mean diffuse lighting
        outputs["diffuse_light"] = torch.clamp(
            linear_to_srgb(torch.mean(diffuse_lights, dim=1)), min=0, max=1
        )

        # for vis
        if not self.training and level == 0:
            outputs["squared_roughness"] = self.get_squared_roughness(roughness)
            outputs["real_roughness"] = torch.sqrt(outputs["squared_roughness"])

            for k in [
                "specular_lights",
                "diffuse_lights",
                "specular_lights_real",
                "specular_weights",
                "outer_vis",
            ]:
                data = locals()[k]
                if isinstance(data, list):
                    for ix, item in enumerate(data):
                        self.save_multi_vis_data(
                            outputs, k + f"_{ix + 1}", item, self.cfg.max_vis_ray_num
                        )
                elif isinstance(data, Mapping):
                    for key, item in data.items():
                        self.save_multi_vis_data(
                            outputs, k + f"_{key}", item, self.cfg.max_vis_ray_num
                        )
                else:
                    self.save_multi_vis_data(outputs, k, data, self.cfg.max_vis_ray_num)

        diffuse_colors = torch.clamp(linear_to_srgb(diffuse_colors), min=0, max=1)
        specular_colors = torch.clamp(linear_to_srgb(specular_colors), min=0, max=1)
        outputs["diffuse_color"] = diffuse_colors
        outputs["specular_color"] = specular_colors
        outputs["approximate_light"] = torch.clamp(
            linear_to_srgb(
                torch.mean(kd[:, :diffuse_num] * diffuse_lights, dim=1) + specular_colors
            ),
            min=0,
            max=1,
        )
        return colors, outputs

    @staticmethod
    def save_multi_vis_data(outputs, key, data, max_vis_num):
        assert data.ndim == 3
        data = torch.clamp(linear_to_srgb(data), min=0, max=1)
        data = data[:, -max_vis_num:, :]
        for i in range(data.shape[1]):
            # if data[:, i, :].shape[0] == 0:
            curr_data = data[:, i, :]
            # else:
            #     curr_data = normalize_using_percentile(data[:, i, :])
            # max_val = curr_data.max() if curr_data.shape[0] != 0 else 1
            # outputs[f"{key}_{i+1}_norm"] = curr_data / max_val
            outputs[f"{key}_{i+1}"] = curr_data

    def forward(self, pts, view_dirs, normals, human_poses, step, is_train):
        # update occ grid for bg nerf sampling
        # if self.training and isinstance(self.bg_model, NeRFRenderer):
        #     assert self.bg_model.use_occ_grid
        #     self.bg_model.update_estimator_nerf(step)

        self.step = step
        view_dirs, normals = F.normalize(view_dirs, dim=-1), F.normalize(normals, dim=-1)
        reflections = calc_ref_dir(view_dirs, normals)  # [pn,3]
        metallic, roughness, albedo = self.predict_materials(pts)  # [pn,1] [pn,1] [pn,3]
        return self.shade_mixed(
            pts, normals, view_dirs, reflections, metallic, roughness, albedo, human_poses, is_train
        )

    # def env_light(self, h, w, gamma=True):
    #     azs = torch.linspace(1.0, 0.0, w) * np.pi * 2 - np.pi / 2
    #     els = torch.linspace(1.0, -1.0, h) * np.pi / 2

    #     els, azs = torch.meshgrid(els, azs)
    #     if self.cfg.is_real:
    #         x = torch.cos(els) * torch.cos(azs)
    #         y = torch.cos(els) * torch.sin(azs)
    #         z = torch.sin(els)
    #     else:
    #         z = torch.cos(els) * torch.cos(azs)
    #         x = torch.cos(els) * torch.sin(azs)
    #         y = torch.sin(els)
    #     xyzs = torch.stack([x, y, z], -1)  # h,w,3
    #     xyzs = xyzs.reshape(h * w, 3)
    #     # xyzs = xyzs @ torch.from_numpy(np.asarray([[0,0,1],[0,1,0],[-1,0,0]],np.float32)).cuda()

    #     batch_size = 8192
    #     lights = []
    #     for ri in range(0, h * w, batch_size):
    #         with torch.no_grad():
    #             light = self.predict_outer_lights_pts(xyzs[ri : ri + batch_size])
    #         lights.append(light)
    #     if gamma:
    #         lights = linear_to_srgb(torch.cat(lights, 0)).reshape(h, w, 3)
    #     else:
    #         lights = (torch.cat(lights, 0)).reshape(h, w, 3)
    #     return lights

    # def predict_outer_lights_pts(self, pts):
    #     if self.cfg.outer_light_version == "direction":
    #         return self.outer_light(self.sph_enc(pts, 0))
    #     elif self.cfg.outer_light_version == "sphere_direction":
    #         return self.outer_light(torch.cat([self.sph_enc(pts, 0), self.sph_enc(pts, 0)], -1))
    #     else:
    #         raise NotImplementedError

    # def get_env_light(self):
    #     return self.predict_outer_lights_pts(self.light_pts)

    def material_regularization(self, pts, normals, metallic, roughness, albedo, step):
        # here roughness is squared!
        # metallic, roughness, albedo = self.predict_materials(pts)
        reg = None

        if self.cfg.reg_change:
            if reg is None:
                reg = 0
            normals = F.normalize(normals, dim=-1)
            x = self.get_orthogonal_directions(normals)
            y = torch.cross(normals, x)
            ang = torch.rand(pts.shape[0], 1) * np.pi * 2
            if self.cfg.change_type == "constant":
                change = (torch.cos(ang) * x + torch.sin(ang) * y) * self.cfg.change_eps
            elif self.cfg.change_type == "gaussian":
                eps = torch.normal(mean=0.0, std=self.cfg.change_eps, size=[x.shape[0], 1])
                change = (torch.cos(ang) * x + torch.sin(ang) * y) * eps
            else:
                raise NotImplementedError
            m0, r0, a0 = self.predict_materials(pts + change)
            reg = reg + torch.mean(
                (
                    torch.abs(m0 - metallic) * self.cfg.reg_scales[0]
                    + torch.abs(r0 - roughness) * self.cfg.reg_scales[1]
                    + torch.abs(a0 - albedo) * self.cfg.reg_scales[2]
                )
                * self.cfg.reg_lambda1,
                dim=1,
            )

        if self.cfg.reg_min_max and step is not None and step < 2000:
            # sometimes the roughness and metallic saturate with the sigmoid activation in the early stage
            if reg is None:
                reg = 0

            max_r, min_r = 0.98, 0.02
            # if self.cfg.use_square_roughness:
            #     max_r, min_r = max_r**2, min_r**2

            reg = reg + torch.sum(torch.clamp(roughness - max_r, min=0))
            reg = reg + torch.sum(torch.clamp(min_r - roughness, min=0))
            reg = reg + torch.sum(torch.clamp(metallic - 0.98, min=0))
            reg = reg + torch.sum(torch.clamp(0.02 - metallic, min=0))

        return reg
