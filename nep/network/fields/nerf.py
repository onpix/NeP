import torch.nn.functional as F
from nerfstudio.fields.nerfacto_field import SHEncoding
from nep.utils.ref_utils import generate_ide_fn, calc_ref_dir

# from nep.ns.models.fakenerf360 import MyNeRFEncoding
from typing import Dict, Optional, Tuple, Type
from nep.network.fields.mipnerf360 import expected_sin
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
import torch.nn as nn
import torch
from jaxtyping import Float
from torch import Tensor
from nerfstudio.utils.math import Gaussians
from nep.network.fields.utils import make_predictor, get_embedder

# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch


class MyNeRFEncoding(NeRFEncoding):
    ...
    # def pytorch_fwd(
    #     self,
    #     in_tensor: Float[Tensor, "*bs input_dim"],
    #     covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None,
    # ) -> Float[Tensor, "*bs output_dim"]:
    #     """
    #     From father, but compute the last dim manually, as the shape of ray could be 0.
    #     """
    #     scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
    #     freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(
    #         in_tensor.device
    #     )
    #     scaled_inputs = scaled_in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]

    #     last_dim = scaled_inputs.shape[-1] * scaled_inputs.shape[-2]
    #     scaled_inputs = scaled_inputs.view(
    #         *scaled_inputs.shape[:-2], last_dim
    #     )  # [..., "input_dim" * "num_scales"]

    #     if covs is None:
    #         encoded_inputs = torch.sin(
    #             torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1)
    #         )
    #     else:
    #         input_var = torch.diagonal(covs, dim1=-2, dim2=-1)[..., :, None] * freqs[None, :] ** 2
    #         last_dim = input_var.shape[-1] * input_var.shape[-2]
    #         input_var = input_var.reshape((*input_var.shape[:-2], last_dim))
    #         encoded_inputs = expected_sin(
    #             torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1),
    #             torch.cat(2 * [input_var], dim=-1),
    #         )

    #     if self.include_input:
    #         encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
    #     return encoded_inputs


class NeRFNetwork(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        d_in=3,
        d_in_view=3,
        multires=0,
        multires_view=0,
        skips=[4],
        use_viewdirs=False,
        density_only=False,
        ns=False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_refnerf=False,
        norm_360_input: bool = True,
        use_sh: bool = False,
    ):
        super(NeRFNetwork, self).__init__()
        self.D = D
        self.W = W
        assert d_in in [3, 4]
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None
        self.mip = ns
        self.spatial_distortion = spatial_distortion
        self.use_refnerf = use_refnerf
        self.density_only = density_only
        self.norm_360_input = norm_360_input

        if multires > 0:
            if not self.mip:
                self.embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
                self.input_ch = input_ch
            else:
                self.embed_fn = MyNeRFEncoding(
                    in_dim=d_in,
                    num_frequencies=multires,
                    min_freq_exp=0.0,
                    max_freq_exp=multires,
                    include_input=True,
                )
                self.input_ch = self.embed_fn.get_out_dim()

        if multires_view > 0:
            if self.use_refnerf:
                print("Use IDE to encode input dirs")
                func = generate_ide_fn(4)
                self.embed_fn_view = lambda x: func(x, 0)
                self.input_ch_view = 38

            elif use_sh:
                print("Use SHEncoding for nerf.")
                self.embed_fn_view = SHEncoding(4, "torch")
                self.input_ch_view = self.embed_fn_view.get_out_dim()

            elif not self.mip:
                self.embed_fn_view, input_ch_view = get_embedder(
                    multires_view, input_dims=d_in_view
                )
                self.input_ch_view = input_ch_view

            else:
                self.embed_fn_view = MyNeRFEncoding(
                    in_dim=d_in_view,
                    num_frequencies=multires_view,
                    min_freq_exp=0.0,
                    max_freq_exp=multires_view,
                    include_input=True,
                )
                self.input_ch_view = self.embed_fn_view.get_out_dim()

        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)]
            + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)
            ]
        )
        self.density_linear = nn.Linear(W, 1)
        if self.density_only:
            print("Init density_only nerf, this network should only be used for proposal network.")
            return

        # Implementation according to the official code release
        # (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        mlp_head_ch = self.input_ch_view + W if use_viewdirs else W
        self.views_linears = nn.ModuleList([nn.Linear(mlp_head_ch, W // 2)])

        if self.use_refnerf:
            self.normal_linear = nn.Sequential(
                nn.Linear(W, W),
                nn.ReLU(),
                nn.Linear(W, W),
                nn.ReLU(),
                nn.Linear(W, 3),
            )

        # Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        self.feature_linear = nn.Linear(W, W)
        self.rgb_linear = nn.Linear(W // 2, 3)

    def get_normals(self, xyz, density_before_activation) -> Float[Tensor, "*batch 3"]:
        """Computes and returns a tensor of normals.

        Args:
            density: Tensor of densities.
        """
        with torch.enable_grad():
            grads = torch.autograd.grad(
                density_before_activation,
                xyz,
                grad_outputs=torch.ones_like(density_before_activation),
                retain_graph=True,
            )[0]

            # flip the directions of gradients, as grads pointing from smaller to larger density
            # the normal direction of a density point should be pointing from larger to smaller density
            normals = -F.normalize(grads, dim=-1)
        return normals

    def forward(self, xyz, dirs, cov=None, density_only=False, normals=None, rgb_only=False):
        """
        Args:
            density_only: if True, only return density
            calc_normal: if True, calculate normal based on density
            dirs: direction form point to camera. please be noted that this direction is flipped from the ray direction.
        """

        if not density_only and self.density_only:
            raise ValueError("density_only is False, but self.density_only is True")

        # if self.embed_fn is not None:
        assert self.embed_fn is not None and self.embed_fn_view is not None
        res = {}
        self._xyz = xyz
        if self.use_refnerf and not self._xyz.requires_grad:
            self._xyz.requires_grad = True

        # hack: warning: hard code here!!!! remember to remove this.
        # cov = None

        # with torch.set_grad_enabled(self.training or self.use_refnerf):
        # prepare inputs
        if cov is not None:
            # contract gaussians, input 3 dims
            if self.spatial_distortion is not None:
                gaussians = self.spatial_distortion(Gaussians(xyz, cov))
                xyz, cov = gaussians.mean, gaussians.cov
                xyz = self.embed_fn(xyz, cov)

                # here after the contraction and PE, the value range of input_pts is [-2, 2]
                # todo: this will worse nerf rgb?
                if self.norm_360_input:
                    xyz = (xyz + 2) / 4.0
            else:
                # note: value range is [-inf, inf] here.
                xyz = self.embed_fn(xyz, cov)
        else:
            # when using cov (mipnerf), the dim of points (mean) should be 3.
            if self.d_in == 4:
                norm = torch.norm(xyz, dim=-1, keepdim=True)
                xyz = torch.cat([xyz / norm, 1.0 / norm], -1)

            elif self.spatial_distortion is not None:
                assert self.d_in == 3
                xyz = self.spatial_distortion(xyz)

            # input 3 or 4 dims
            xyz = self.embed_fn(xyz)

        # forward backbone
        h = xyz
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([xyz, h], -1)

        # res["density"] = F.relu(self.density_linear(h))
        if not rgb_only:
            res["density"] = self.density_linear(h)

        if density_only:
            return res

        feature = self.feature_linear(h)

        # predict normal
        if self.use_refnerf:
            grads_pred = self.normal_linear(feature)
            res["pred_normal"] = -F.normalize(grads_pred, dim=-1)
            if self.training:
                res["grad_normal"] = self.get_normals(self._xyz, res["density"])
            else:
                res["grad_normal"] = torch.ones_like(res["pred_normal"])

        if not density_only and self.embed_fn_view is not None:
            if self.use_refnerf:
                if normals is not None:
                    normals_to_use = normals
                else:
                    normals_to_use = res["pred_normal"]
                if self.use_viewdirs:
                    dirs = calc_ref_dir(dirs, normals_to_use)

            if self.use_viewdirs:
                dirs = self.embed_fn_view(dirs)
                # after PE, the value range of input_views is [-1, 1]
                dirs = (dirs + 1) / 2.0

        # predict rgb
        if self.use_viewdirs:
            h = torch.cat([feature, dirs], -1)
        else:
            h = feature

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)
        res["rgb"] = self.rgb_linear(h)
        return res
