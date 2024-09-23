import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from nep.network.fields.utils import make_predictor, get_embedder


class SDFNetwork(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        skip_in=(4,),
        multires=0,
        bias=0.5,
        scale=1,
        geometric_init=True,
        weight_norm=True,
        inside_outside=False,
        sdf_activation="none",
        layer_activation="softplus",
    ):
        super(SDFNetwork, self).__init__()

        # from nerfstudio.fields.sdf_field import SDFField, SDFFieldConfig
        # self.field = SDFField(SDFFieldConfig(), num_images=89, aabb=torch.tensor([[-1, -1, -1], [1, 1, 1]]), use_average_appearance_embedding=False)
        # return

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            dims[0] = input_ch

            # Option 2: using ns
            # from nerfstudio.field_components.encodings import NeRFEncoding
            # embed_fn = NeRFEncoding(
            #     in_dim=d_in, num_frequencies=multires, min_freq_exp=0.0, max_freq_exp=16.0, include_input=True
            # )
            # dims[0] = embed_fn.get_out_dim()

            self.embed_fn_fine = embed_fn

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=-np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if layer_activation == "softplus":
            self.activation = nn.Softplus(beta=100)
        elif layer_activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError

    def _forward(self, points, cov):
        # check_memory_usage(f'sdf: before PE | {with_grad}')
        assert self.embed_fn_fine is not None and cov is None
        inputs = self.embed_fn_fine(points)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return x

    def forward(self, points, cov=None, with_grad=False, use_numerical_grad=False):
        """
        output is a [..., 257] feature, where the first element is sdf.
        """
        # get features
        if with_grad and not use_numerical_grad:
            points.requires_grad_(True)
            with torch.enable_grad():
                x = self._forward(points, cov)
        else:
            x = self._forward(points, cov)

        # return feature if grad is not needed
        if not with_grad:
            return x

        # get grad
        if use_numerical_grad:
            delta = 0.0001
            neighbor_points = torch.stack(
                [
                    points + torch.as_tensor([delta, 0.0, 0.0]).to(x),
                    points + torch.as_tensor([-delta, 0.0, 0.0]).to(x),
                    points + torch.as_tensor([0.0, delta, 0.0]).to(x),
                    points + torch.as_tensor([0.0, -delta, 0.0]).to(x),
                    points + torch.as_tensor([0.0, 0.0, delta]).to(x),
                    points + torch.as_tensor([0.0, 0.0, -delta]).to(x),
                ],
                dim=0,
            )

            points_sdf = self._forward(neighbor_points, cov=None)[..., 1]
            gradients = torch.stack(
                [
                    0.5 * (points_sdf[0] - points_sdf[1]) / delta,
                    0.5 * (points_sdf[2] - points_sdf[3]) / delta,
                    0.5 * (points_sdf[4] - points_sdf[5]) / delta,
                ],
                dim=-1,
            )
        else:
            # use pytorch to get grad
            with torch.enable_grad():
                sdf = x[..., :1]
                d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
                gradients = torch.autograd.grad(
                    outputs=sdf,
                    inputs=points,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]

        # check_memory_usage(f'sdf: after grad | {with_grad}')
        return x, gradients

    def sdf(self, x):
        return self.forward(x, with_grad=False)[..., :1]

    # def forward(self, inputs, with_grad=False):
    #     from nerfstudio.field_components.field_heads import FieldHeadNames
    #     shape = inputs.shape
    #     inputs = inputs.view(-1, 3)
    #     inputs.requires_grad_(True)
    #     with torch.enable_grad():
    #         hidden_output = self.field.forward_geonetwork(inputs)
    #         sdf, geo_feature = torch.split(hidden_output, [1, self.field.config.geo_feat_dim], dim=-1)
    #     d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
    #     gradients = torch.autograd.grad(
    #         outputs=sdf, inputs=inputs, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
    #     )[0]

    #     # rgb = self.get_colors(inputs, directions_flat, gradients, geo_feature, camera_indices)

    #     # rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
    #     sdf = sdf.view(*shape[:-1], -1)
    #     gradients = gradients.view(*shape[:-1], -1)
    #     normals = torch.nn.functional.normalize(gradients, p=2, dim=-1)

    #     outputs = {}
    #     outputs.update(
    #         {
    #             # FieldHeadNames.RGB: rgb,
    #             FieldHeadNames.SDF: sdf,
    #             FieldHeadNames.NORMALS: normals,
    #             FieldHeadNames.GRADIENT: gradients,
    #         }
    #     )
    #     if with_grad:
    #         return hidden_output.view(*shape[:-1], -1), gradients
    #     else:
    #         return hidden_output.view(*shape[:-1], -1)

    # def sdf_hidden_appearance(self, x):
    #     return self.forward(x)

    # def gradient(self, x):
    #     x.requires_grad_(True)
    #     with torch.enable_grad():
    #         y = self.sdf(x)
    #     d_output = torch.ones_like(y, requires_grad=False, device=y.device)
    #     gradients = torch.autograd.grad(
    #         outputs=y,
    #         inputs=x,
    #         grad_outputs=d_output,
    #         create_graph=True,
    #         retain_graph=True,
    #         only_inputs=True)[0]
    #     return gradients

    # def sdf_normal(self, x):
    #     x.requires_grad_(True)
    #     with torch.enable_grad():
    #         y = self.sdf(x)
    #     d_output = torch.ones_like(y, requires_grad=False, device=y.device)
    #     gradients = torch.autograd.grad(
    #         outputs=y,
    #         inputs=x,
    #         grad_outputs=d_output,
    #         create_graph=True,
    #         retain_graph=True,
    #         only_inputs=True)[0]
    #     return y[...,:1].detach(), gradients.detach()


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val, activation="exp"):
        super(SingleVarianceNetwork, self).__init__()
        self.act = activation
        self.register_parameter("variance", nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        if self.act == "exp":
            return torch.ones([*x.shape[:-1], 1]) * torch.exp(self.variance * 10.0)
        elif self.act == "linear":
            return torch.ones([*x.shape[:-1], 1]) * self.variance * 10.0
        elif self.act == "square":
            return torch.ones([*x.shape[:-1], 1]) * (self.variance * 10.0) ** 2
        else:
            raise NotImplementedError

    def warp(self, x, inv_s):
        return torch.ones([*x.shape[:-1], 1]) * inv_s
