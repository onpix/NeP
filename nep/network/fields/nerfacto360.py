from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.field_components.field_heads import FieldHeadNames
from jaxtyping import Float, Shaped
from nerfstudio.fields.base_field import Field
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.cameras.rays import RaySamples
import torch
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from typing import Dict, Literal, Optional, Tuple
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from torch import Tensor, nn


def samples_to_360_encoding(
    ray_samples, spatial_distortion, encoding, norm_gaussians=True
):
    assert spatial_distortion is not None
    gaussian_samples = ray_samples.frustums.get_gaussian_blob()
    gaussian_samples = spatial_distortion(gaussian_samples)

    # normalize gaussian samples
    if norm_gaussians:
        mean = (gaussian_samples.mean + 2) / 4
        cov = gaussian_samples.cov / 16
    else:
        mean, cov = gaussian_samples.mean, gaussian_samples.cov

    encoded_xyz = encoding(mean, covs=cov)
    return encoded_xyz


class ProposalField(Field):
    """
    This field should be only used for proposal network: because this model will be called as:
        density_fn[i](ray_samples.frustums.get_positions())
    """

    def __init__(
        self,
        num_layers: int = 4,
        hidden_dim: int = 128,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_linear: bool = False,
        norm_gaussians: bool = True,
        append_xyz_norm: bool = False,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()
        self.spatial_distortion = spatial_distortion
        self.use_linear = use_linear
        self.norm_gaussians = norm_gaussians
        self.encoding = NeRFEncoding(
            in_dim=3 if not append_xyz_norm else 4,
            num_frequencies=16,
            min_freq_exp=0.0,
            max_freq_exp=16.0,
            include_input=True,
        )
        self.mlp_in_dim = self.encoding.get_out_dim()
        if not self.use_linear:
            self.mlp_base = MLP(
                in_dim=self.mlp_in_dim,
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=1,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
        else:
            self.linear = torch.nn.Linear(self.mlp_in_dim, 1)

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, None]:
        encoded_xyz = samples_to_360_encoding(
            ray_samples, self.spatial_distortion, self.encoding, self.norm_gaussians
        )

        if not self.use_linear:
            density_before_activation = (
                self.mlp_base(encoded_xyz.view(-1, self.mlp_in_dim))
                .view(*ray_samples.frustums.shape, -1)
                .to(encoded_xyz)
            )
        else:
            density_before_activation = self.linear(encoded_xyz).view(
                *ray_samples.frustums.shape, -1
            )

        density = trunc_exp(density_before_activation)
        # density = torch.nn.functional.softplus(density_before_activation)
        return density, None


class Nerfacto360Field(NerfactoField):
    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        skip_connections: Tuple[int] = (4,),
        num_layers: int = 8,
        hidden_dim: int = 128,
        geo_feat_dim: int = 256,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 2,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 128,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        norm_gaussians: bool = True,
        disable_appearance_embedding: bool = False,
        append_xyz_norm: bool = False,
    ) -> None:
        super().__init__(
            aabb,
            num_images,
            num_layers,
            hidden_dim,
            geo_feat_dim,
            num_levels,
            base_res,
            max_res,
            log2_hashmap_size,
            num_layers_color,
            num_layers_transient,
            features_per_level,
            hidden_dim_color,
            hidden_dim_transient,
            appearance_embedding_dim,
            transient_embedding_dim,
            use_transient_embedding,
            use_semantics,
            num_semantic_classes,
            pass_semantic_gradients,
            use_pred_normals,
            use_average_appearance_embedding,
            spatial_distortion,
            implementation,
        )

        # update IPE
        self.position_encoding = NeRFEncoding(
            in_dim=3 if not append_xyz_norm else 4,
            num_frequencies=16,
            min_freq_exp=0.0,
            max_freq_exp=16.0,
            include_input=True,
        )
        # self.direction_encoding = NeRFEncoding(in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4,
        #                                        include_input=True)
        # always use tcnn sh encoding
        # self.direction_encoding = SHEncoding(levels=4, implementation='tcnn')

        # use nerf mlp without hash grid
        del self.mlp_base_grid
        assert implementation == "torch"

        # remake mlp_base; mlp_head is the same as father
        self.mlp_in_dim = self.position_encoding.get_out_dim()
        self.norm_gaussians = norm_gaussians
        self.disable_appearance_embedding = disable_appearance_embedding
        self.mlp_base = MLP(
            in_dim=self.mlp_in_dim,
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            implementation=implementation,
            skip_connections=skip_connections,
        )

        # remake mlp_head
        mlp_head_in_dim = self.direction_encoding.get_out_dim() + self.geo_feat_dim
        if not self.disable_appearance_embedding:
            print("Use appearance embedding for nerfacto360 field.")
            mlp_head_in_dim += self.appearance_embedding_dim
        else:
            print("Do not use appearance embedding for nerfacto360 field.")

        # todo: if remake mlp_head here, there will be a mysterious bug: can not see object in the results. wtf???
        # self.mlp_head = MLP(
        #     in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
        #     num_layers=num_layers_color,
        #     layer_width=hidden_dim_color,
        #     out_dim=3,
        #     activation=nn.ReLU(),
        #     out_activation=nn.Sigmoid(),
        #     implementation=implementation,
        # )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        # todo: should we apply selector here?
        encoded_xyz = samples_to_360_encoding(
            ray_samples,
            self.spatial_distortion,
            self.position_encoding,
            norm_gaussians=self.norm_gaussians,
        )

        h = self.mlp_base(encoded_xyz.view(-1, self.mlp_in_dim)).view(
            *ray_samples.frustums.shape, -1
        )
        density_before_activation, base_mlp_out = torch.split(
            h, [1, self.geo_feat_dim], dim=-1
        )
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(encoded_xyz))
        # density = torch.nn.functional.softplus(density_before_activation)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        # here direction is from -1 to 1
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        d = self.direction_encoding(directions.view(-1, 3))
        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        if not self.disable_appearance_embedding:
            # use appearance embedding
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                if self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim),
                        device=directions.device,
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim),
                        device=directions.device,
                    )
            h = torch.cat(
                [
                    d,
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_appearance.view(-1, self.appearance_embedding_dim),
                ],
                dim=-1,
            )
        else:
            # remove appearance embedding here.
            # h = torch.cat([d, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)
            h = torch.cat(
                [
                    d,
                    density_embedding.view(-1, self.geo_feat_dim),
                    torch.zeros(
                        (d.shape[0], self.appearance_embedding_dim),
                        device=directions.device,
                    ),
                ],
                dim=-1,
            )

        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})
        return outputs
