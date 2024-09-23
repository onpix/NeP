from nerfstudio.cameras.cameras import Cameras, CameraType
import yaml
from nerfstudio.configs.method_configs import all_methods
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from pathlib import Path
from typing import Literal, Optional, Tuple
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
import torch
from nerfstudio.utils.rich_utils import CONSOLE
import os, sys
import nerfstudio.cameras.cameras
import nerfstudio.cameras.camera_paths
import nerfstudio.data.dataparsers.colmap_dataparser
import nerfstudio.utils.eval_utils as eval_utils
from nerfstudio.viewer.server import viewer_state
# from nerfstudio.scripts.viewer import run_viewer
from typing import Dict, List, Literal, Optional, Tuple, Union
from jaxtyping import Float, Int, Shaped
from .nep_dataparser import get_human_poses
from torch import Tensor
import nep


class MyCameras(Cameras):
    def __init__(
        self,
        camera_to_worlds: Float[Tensor, "*batch_c2ws 3 4"],
        fx: Union[Float[Tensor, "*batch_fxs 1"], float],
        fy: Union[Float[Tensor, "*batch_fys 1"], float],
        cx: Union[Float[Tensor, "*batch_cxs 1"], float],
        cy: Union[Float[Tensor, "*batch_cys 1"], float],
        width: Optional[Union[Shaped[Tensor, "*batch_ws 1"], int]] = None,
        height: Optional[Union[Shaped[Tensor, "*batch_hs 1"], int]] = None,
        distortion_params: Optional[Float[Tensor, "*batch_dist_params 6"]] = None,
        camera_type: Union[
            Int[Tensor, "*batch_cam_types 1"],
            int,
            List[CameraType],
            CameraType,
        ] = CameraType.PERSPECTIVE,
        times: Optional[Float[Tensor, "num_cameras"]] = None,
        metadata: Optional[Dict] = None,
        # this is the only method to switch opencv / opengl for cameras:
        # use_opencv_system: bool = True
        use_opencv_system: bool = False,
    ) -> None:
        if metadata is None:
            metadata = {}
        metadata["human_poses"] = get_human_poses(camera_to_worlds, False).view(-1, 12)
        self.use_opencv_system = use_opencv_system
        # print('Use custom cameras to override ns')

        super().__init__(
            camera_to_worlds,
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            distortion_params,
            camera_type,
            times,
            metadata,
        )

nerfstudio.cameras.cameras.Cameras = MyCameras
nerfstudio.cameras.camera_paths.Cameras = MyCameras
nerfstudio.data.dataparsers.colmap_dataparser.Cameras = MyCameras
nep.ns.nep_dataparser.Cameras = MyCameras
viewer_state.Cameras = MyCameras


