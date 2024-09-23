import torch.nn.functional as F
import ipdb
import torch.nn as nn
import torch
import numpy as np
import nvdiffrast.torch as dr

from nep.utils.raw_utils import linear_to_srgb
from nep.utils.ref_utils import generate_ide_fn, calc_ref_dir
from nep.utils.base_utils import roughness2area, merge_config
from nerfstudio.cameras.rays import RayBundle
from typing import Any
import omegaconf
from nerfstudio.field_components.encodings import SHEncoding
from nep.network.fields.nerf import NeRFNetwork
from nep.network.fields.utils import (
    make_predictor,
    get_embedder,
    get_camera_plane_intersection,
    get_sphere_intersection,
    offset_points_to_sphere,
    IPE,
)


class AppShadingNetwork(nn.Module):
    default_cfg = {}

    def __init__(self, cfg, outer_nerf_fn=None):
        super().__init__()
        # self.cfg = {**self.default_cfg, **cfg}
        # self.cfg: Any = merge_config(self.default_cfg, cfg)
        self.cfg: Any = omegaconf.OmegaConf.create(cfg)
        self.use_bg_model = self.cfg.get("use_bg_model", False)
        self.use_refneus = self.cfg.use_refneus
        feats_dim = 256
        self.use_ns_sh = False
        if self.cfg.d_enc == "sh":
            self.sph_enc = SHEncoding(4, "torch")
            ide_dim = self.sph_enc.get_out_dim()
        elif self.cfg.d_enc == "ide":
            self.sph_enc = generate_ide_fn(4)
            ide_dim = 38
        elif self.cfg.d_enc == "pe":
            self.sph_enc, ide_dim = get_embedder(4, 3)
        else:
            raise NotImplementedError

        create_mlp = lambda D, W: NeRFNetwork(
            D=D,
            d_in=4,
            d_in_view=3,
            W=W,
            multires=10,
            multires_view=4,
            skips=[4],
            use_viewdirs=True,
            density_only=False,
            ns=False,
            spatial_distortion=None,
            use_refnerf=False,
        )

        # human lights are the lights reflected from the photo capturer
        if self.cfg.human_light:
            self.human_light_predictor = make_predictor(2 * 2 * 6, 4, activation=self.cfg.light_act)
            nn.init.constant_(self.human_light_predictor[-2].bias, np.log(0.01))

        # roughness MLP
        self.roughness_predictor = make_predictor(feats_dim + 3, 1)
        if self.cfg.roughness_init != 0:
            nn.init.constant_(self.roughness_predictor[-2].bias, self.cfg.roughness_init)

        # for ref neus
        if self.use_refneus:
            print("Init App shading network: use refneus")
            act = "sigmoid" if not self.cfg.release else "relu1.5"
            self.color_mlp_nodir = make_predictor(feats_dim + 3, 3, activation=act)
            # nn.init.constant_(self.color_mlp_nodir[-2].bias, np.log(0.5))
            if not self.cfg.refneus_with_app:
                self.color_mlp_withdir = make_predictor(feats_dim + ide_dim, 3, activation=act)
                return

        # material MLPs
        self.metallic_predictor = make_predictor(feats_dim + 3, 1)
        if self.cfg.metallic_init != 0:
            nn.init.constant_(self.metallic_predictor[-2].bias, self.cfg.metallic_init)
        self.albedo_predictor = make_predictor(feats_dim + 3, 3)

        FG_LUT = torch.from_numpy(
            np.fromfile("nep/assets/bsdf_256_256.bin", dtype=np.float32).reshape(1, 256, 256, 2)
        )
        self.register_buffer("FG_LUT", FG_LUT)

        self.dir_enc, dir_pe_dim = get_embedder(6, 3)
        self.pos_enc, pos_dim = get_embedder(self.cfg.light_pos_freq, 3)
        exp_max = self.cfg.light_exp_max
        # outer lights are direct lights
        if self.use_bg_model:
            self.outer_nerf_fn = outer_nerf_fn
            self.diffuse_light = make_predictor(
                feats_dim + 3, 3, activation=self.cfg.light_act, exp_max=exp_max
            )
            self.roughness_transform = eval(self.cfg.roughness_transform)
        else:
            if self.cfg.sphere_direction:
                self.outer_light = make_predictor(
                    ide_dim * 2, 3, activation=self.cfg.light_act, exp_max=exp_max
                )
            else:
                self.outer_light = make_predictor(
                    ide_dim, 3, activation=self.cfg.light_act, exp_max=exp_max
                )
            nn.init.constant_(self.outer_light[-2].bias, np.log(0.5))

        # inner lights are indirect lights
        if self.cfg.deep_inner_light_mlp:
            self.inner_light = create_mlp(4, 128)
        else:
            self.inner_light = make_predictor(
                pos_dim + ide_dim, 3, activation=self.cfg.light_act, exp_max=exp_max
            )
            nn.init.constant_(self.inner_light[-2].bias, np.log(0.5))

        self.inner_weight = make_predictor(pos_dim + dir_pe_dim, 1, activation="none")
        nn.init.constant_(self.inner_weight[-2].bias, self.cfg.inner_init)

    def predict_human_light(self, points, reflective, human_poses, roughness):
        inter, dists, hits = get_camera_plane_intersection(points, reflective, human_poses)
        scale_factor = 0.3
        mean = inter[..., :2] * scale_factor
        var = roughness * (dists[:, None] * scale_factor) ** 2
        hits = hits & (torch.norm(mean, dim=-1) < 1.5) & (dists > 0)
        hits = hits.float().unsqueeze(-1)
        mean = mean * hits
        var = var * hits

        var = var.expand(mean.shape[0], 2)
        pos_enc = IPE(mean, var, 0, 6)  # 2*2*6
        human_lights = self.human_light_predictor(pos_enc)
        human_lights = human_lights * hits
        human_lights, human_weights = human_lights[..., :3], human_lights[..., 3:]
        human_weights = torch.clamp(human_weights, max=1.0, min=0.0)
        return human_lights, human_weights

    def predict_specular_lights(
        self, points, feature_vectors, reflective, roughness, human_poses, step
    ):
        human_light, human_weight = 0, 0
        ref_roughness = self.sph_enc(reflective, roughness)

        pts = self.pos_enc(points)
        if self.use_bg_model:
            raise NotImplementedError
            pixel_area = roughness2area(
                roughness,
                max_tan=self.cfg.max_tan,
                roughness_transform=self.roughness_transform,
            )
            raybundle = RayBundle(
                origins=points, directions=reflective, nears=0.1, fars=100.0, pixel_area=pixel_area
            )
            assert self.outer_nerf_fn is not None
            direct_light = self.outer_nerf_fn(raybundle)
        else:
            if self.cfg.sphere_direction:
                sph_points = offset_points_to_sphere(points)
                sph_points = F.normalize(
                    sph_points + reflective * get_sphere_intersection(sph_points, reflective),
                    dim=-1,
                )
                sph_points = self.sph_enc(sph_points, roughness)
                direct_light = self.outer_light(torch.cat([ref_roughness, sph_points], -1))
            else:
                # why modified: convert the dtype of SH output to float32
                direct_light = self.outer_light(ref_roughness)

        if self.cfg.human_light and human_poses is not None:
            human_light, human_weight = self.predict_human_light(
                points, reflective, human_poses, roughness
            )
        else:
            human_light = 0
            human_weight = 0

        if self.cfg.deep_inner_light_mlp:
            indirect_light = self.inner_light(points, reflective, rgb_only=True)["rgb"]
        else:
            indirect_light = self.inner_light(torch.cat([pts, ref_roughness], -1))

        ref_ = self.dir_enc(reflective)
        # why modified: convert the dtype of SH output to float32
        occ_prob = self.inner_weight(
            torch.cat([pts.detach(), ref_.detach()], -1)
        )  # this is occlusion prob
        occ_prob = occ_prob * 0.5 + 0.5
        occ_prob_ = torch.clamp(occ_prob, min=0, max=1)

        light = indirect_light * occ_prob_ + (
            human_light * human_weight + direct_light * (1 - human_weight)
        ) * (1 - occ_prob_)
        indirect_light = indirect_light * occ_prob_
        return light, occ_prob, indirect_light, human_light * human_weight

    def predict_diffuse_lights(self, points, feature, normals):
        roughness = torch.ones([normals.shape[0], 1])
        # todo: remove ide here, as diffuse color is not related to roughness.
        # why modified: use SHEncoding to replace IDE.
        if self.use_ns_sh:
            ref = self.sph_enc(normals).type_as(points)
        else:
            ref = self.sph_enc(normals, roughness)

        if self.cfg.sphere_direction:
            assert not self.use_ns_sh
            sph_points = offset_points_to_sphere(points)
            sph_points = F.normalize(
                sph_points + normals * get_sphere_intersection(sph_points, normals),
                dim=-1,
            )
            sph_points = self.sph_enc(sph_points, roughness)
            light = self.outer_light(torch.cat([ref, sph_points], -1))
        else:
            # why modified: convert the dtype of SH output to float32
            light = self.outer_light(ref)

        return light

    def forward_refneus(
        self, xyz, ref_dir, normals, feature, human_poses, return_inter_results=False
    ):
        extras = {}
        roughness = self.roughness_predictor(torch.cat([feature, xyz], -1))
        if self.cfg.d_enc != "ide":
            dirs_input = self.sph_enc(ref_dir)
        else:
            dirs_input = self.sph_enc(ref_dir, roughness)
        rgb_nodir = self.color_mlp_nodir(torch.cat([feature, xyz], -1))

        if self.cfg.human_light:
            human_light, human_weight = self.predict_human_light(
                xyz, ref_dir, human_poses, roughness
            )
        else:
            human_light = 0
            human_weight = 0

        if self.cfg.refneus_single_mlp:
            rgb = rgb_nodir * (1 - human_weight) + human_light * human_weight

        elif self.cfg.refneus_single_dir_mlp:
            rgb_withdir = self.color_mlp_withdir(torch.cat([feature, dirs_input], -1))
            rgb = rgb_withdir * (1 - human_weight) + human_light * human_weight

        elif self.cfg.lr_decomp:
            L = self.color_mlp_withdir(torch.cat([feature, dirs_input], -1))
            rgb = rgb_nodir * L
            rgb = rgb * (1 - human_weight) + human_light * human_weight
            # rgb = torch.clamp(rgb, min=0.0, max=1.0)
            extras["rgb_nodir"] = rgb_nodir
            extras["L"] = L
            extras["rgb_withdir"] = rgb

        else:
            rgb = None
            if not self.cfg.refneus_with_app:
                rgb_withdir = self.color_mlp_withdir(torch.cat([feature, dirs_input], -1))
                rgb = rgb_withdir * (1 - human_weight) + human_light * human_weight
                extras["rgb_withdir"] = rgb_withdir

            extras["rgb_nodir"] = rgb_nodir

        extras["reflective"] = ref_dir
        inter_results = {
            "roughness": roughness,
        }
        if return_inter_results == True:
            return rgb, extras, inter_results
        else:
            return rgb, extras

    def forward(
        self, points, normals, view_dirs, feature, human_poses, inter_results=False, step=None
    ):
        normals = F.normalize(normals, dim=-1)
        view_dirs = F.normalize(view_dirs, dim=-1)
        reflective = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
        occ_info = {}

        if self.use_refneus:
            refneus_dir = reflective if self.cfg.refneus_with_refdir else view_dirs
            refneus_res = self.forward_refneus(
                points, refneus_dir, normals, feature, human_poses, inter_results
            )
            if self.cfg.refneus_with_app:
                occ_info.update(refneus_res[1])
            else:
                return refneus_res

        NoV = torch.sum(normals * view_dirs, -1, keepdim=True)
        metallic = self.metallic_predictor(torch.cat([feature, points], -1))
        roughness = self.roughness_predictor(torch.cat([feature, points], -1))
        albedo = self.albedo_predictor(torch.cat([feature, points], -1))

        # diffuse light
        # assume a view-independent diffuse light
        diffuse_albedo = (1 - metallic) * albedo

        # if self.use_bg_model:
        #     # diffuse light is related to xyz only
        #     diffuse_light = self.diffuse_light(torch.cat([feature, points], -1))
        # else:

        # note: diffuse light is the normal-direction outer light
        diffuse_light = self.predict_diffuse_lights(points, feature, normals)
        diffuse_color = diffuse_albedo * diffuse_light

        # specular light
        specular_albedo = 0.04 * (1 - metallic) + metallic * albedo
        (
            specular_light,
            occ_prob,
            indirect_light,
            human_light,
        ) = self.predict_specular_lights(points, feature, reflective, roughness, human_poses, step)

        fg_uv = torch.cat(
            [
                torch.clamp(NoV, min=0.0, max=1.0),
                torch.clamp(roughness, min=0.0, max=1.0),
            ],
            -1,
        )
        pn, bn = points.shape[0], 1
        fg_lookup = dr.texture(
            self.FG_LUT,
            fg_uv.reshape(1, pn // bn, bn, -1).contiguous(),
            filter_mode="linear",
            boundary_mode="clamp",
        ).reshape(pn, 2)
        specular_ref = specular_albedo * fg_lookup[:, 0:1] + fg_lookup[:, 1:2]
        specular_color = specular_ref * specular_light

        # integrated together
        color = diffuse_color + specular_color

        # gamma correction
        diffuse_color = linear_to_srgb(diffuse_color)
        specular_color = linear_to_srgb(specular_color)
        color = linear_to_srgb(color)
        color = torch.clamp(color, min=0.0, max=1.0)

        occ_info.update(
            {
                "reflective": reflective,
                "occ_prob": occ_prob,
                "inner_light": indirect_light,
                "outer_light": specular_light,
            }
        )

        if self.cfg.refneus_with_app:
            occ_info["rgb_withdir"] = color

        if inter_results:
            intermediate_results = {
                "specular_albedo": specular_albedo,
                "specular_ref": torch.clamp(specular_ref, min=0.0, max=1.0),
                "specular_light": torch.clamp(linear_to_srgb(specular_light), min=0.0, max=1.0),
                "specular_color": torch.clamp(specular_color, min=0.0, max=1.0),
                "diffuse_albedo": diffuse_albedo,
                "diffuse_light": torch.clamp(linear_to_srgb(diffuse_light), min=0.0, max=1.0),
                "diffuse_color": torch.clamp(diffuse_color, min=0.0, max=1.0),
                "metallic": metallic,
                "roughness": roughness,
                "occ_prob": torch.clamp(occ_prob, max=1.0, min=0.0),
                "indirect_light": indirect_light,
            }
            if self.cfg.human_light:
                intermediate_results["human_light"] = linear_to_srgb(human_light)
            return color, occ_info, intermediate_results
        else:
            return color, occ_info

    def predict_materials(self, points, feature_vectors):
        metallic = self.metallic_predictor(torch.cat([feature_vectors, points], -1))
        roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
        albedo = self.albedo_predictor(torch.cat([feature_vectors, points], -1))
        return metallic, roughness, albedo
