# this line is to override the ns viewer. must be imported at the first line:nsconf
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.models.neus_facto import NeuSFactoModelConfig
from nerfstudio.models.neus import NeuSModelConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig

if True:
    from nep.ns.cameras import MyCameras

    # from nep.ns.override import eval_load_checkpoint

from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.configs.base_config import (
    MachineConfig,
    ViewerConfig,
    LoggingConfig,
    LocalWriterConfig,
)
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    MultiStepSchedulerConfig,
    ExponentialDecaySchedulerConfig,
)
from nep.ns.trainer import MyTrainerConfig

from nep.ns.datamanager import NePDataManagerConfig
from nep.ns.pipeline import NePPipelineConfig
from nep.ns.models.nep import NePModelConfig
from nep.ns.models.nf import MyNerfactoModelConfig
from nep.ns.nep_dataparser import NePColmapDataParserConfig
import numpy as np

np.set_printoptions(precision=6, suppress=True)

nep_method = MethodSpecification(
    config=MyTrainerConfig(
        method_name="nep",
        steps_per_eval_batch=999999,
        steps_per_eval_image=10000,
        steps_per_eval_all_images=999999,
        steps_per_save=2000,
        max_num_iterations=300001,
        mixed_precision=False,
        use_grad_scaler=False,
        pipeline=NePPipelineConfig(
            datamanager=NePDataManagerConfig(
                dataparser=NePColmapDataParserConfig(
                    train_split_fraction=0.95  # only use one eval image
                ),
                train_num_rays_per_batch=512,
                # train_num_rays_per_batch=2048,
                eval_num_rays_per_batch=1024,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                eval_num_images_to_sample_from=1,
            ),
            model=NePModelConfig(),
        ),
        optimizers={
            "network": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-8, weight_decay=0),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=300001),
            },
            "prop": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-8, weight_decay=0),
                "scheduler": CosineDecaySchedulerConfig(max_steps=300001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer+tensorboard",
        logging=LoggingConfig(local_writer=LocalWriterConfig(max_log_size=0)),
        machine=MachineConfig(seed=6033),
    ),
    description="nep stage 1 model",
)

nf_method = MethodSpecification(
    MyTrainerConfig(
        method_name="nf",
        steps_per_eval_batch=500,
        steps_per_eval_image=1000,
        steps_per_save=2000,
        max_num_iterations=30000,
        # mixed_precision=False,
        # use_grad_scaler=False,
        # mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=NePDataManagerConfig(
                dataparser=NePColmapDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                # camera_optimizer=CameraOptimizerConfig(
                #     mode="SO3xR3",
                #     optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                #     scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                # ),
            ),
            model=MyNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer+tensorboard",
        logging=LoggingConfig(local_writer=LocalWriterConfig(max_log_size=0)),
    ),
    description="nerfacto",
)

neusfacto_method = MethodSpecification(
    MyTrainerConfig(
        method_name="nf2",
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=2000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=20001,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=NePDataManagerConfig(
                # _target=VanillaDataManager[SDFDataset],
                # dataparser=SDFStudioDataParserConfig(),
                dataparser=NePColmapDataParserConfig(),
                train_num_rays_per_batch=2048,
                eval_num_rays_per_batch=2048,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                ),
            ),
            model=NeuSFactoModelConfig(
                # proposal network allows for significantly smaller sdf/color network
                sdf_field=SDFFieldConfig(
                    use_grid_feature=True,
                    num_layers=2,
                    num_layers_color=2,
                    hidden_dim=256,
                    bias=0.5,
                    beta_init=0.8,
                    use_appearance_embedding=False,
                ),
                background_model="none",
                eval_num_rays_per_chunk=2048,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": MultiStepSchedulerConfig(
                    max_steps=20001, milestones=(10000, 1500, 18000)
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(
                    warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001
                ),
            },
            "field_background": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(
                    warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer+tensorboard",
    ),
    description="neus-facto",
)
