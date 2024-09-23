import math
import struct
from nerfstudio.data.dataparsers.blender_dataparser import (
    BlenderDataParserConfig,
    Blender,
)
from nerfstudio.utils.io import load_from_json
import imageio
from nerfstudio.utils.colors import get_color
from nerfstudio.data.dataparsers.colmap_dataparser import (
    ColmapDataParserConfig,
    ColmapDataParser,
)
from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils
from nerfstudio.process_data.colmap_utils import parse_colmap_camera_params
from dataclasses import dataclass, field
from pathlib import Path
from dataclasses import dataclass, field
import cv2
from pathlib import Path
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.camera_utils import rotation_matrix
from typing import Type, Optional, Union, Tuple, Literal, List
from jaxtyping import Float
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras, CameraType, CAMERA_MODEL_TO_TYPE
from nerfstudio.data.dataparsers.base_dataparser import (
    DataparserOutputs,
)
import numpy as np
from nerfstudio.data.scene_box import SceneBox
from nep.colmap import plyfile
from nep.utils.base_utils import read_pickle
import torch
import torch.nn.functional as F
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
import collections

Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def normalize_poses(poses, pts, up_est_method, center_est_method):
    poses = poses[:, :3, :]
    if center_est_method == "camera":
        # estimation scene center as the average of all camera positions
        center = poses[..., 3].mean(0)
    elif center_est_method == "lookat":
        # estimation scene center as the average of the intersection of selected pairs of camera rays
        cams_ori = poses[..., 3]
        cams_dir = poses[:, :3, :3] @ torch.as_tensor([0.0, 0.0, -1.0])
        cams_dir = F.normalize(cams_dir, dim=-1)
        A = torch.stack([cams_dir, -cams_dir.roll(1, 0)], dim=-1)
        b = -cams_ori + cams_ori.roll(1, 0)
        t = torch.linalg.lstsq(A, b).solution
        center = (
            torch.stack([cams_dir, cams_dir.roll(1, 0)], dim=-1) * t[:, None, :]
            + torch.stack([cams_ori, cams_ori.roll(1, 0)], dim=-1)
        ).mean((0, 2))

        # cam_mean = poses[..., 3].mean(0)
        # cam_mean[-1] = center[-1]
        # center = cam_mean

    elif center_est_method == "point":
        # first estimation scene center as the average of all camera positions
        # later we'll use the center of all points bounded by the cameras as the final scene center
        center = poses[..., 3].mean(0)
    else:
        raise NotImplementedError(f"Unknown center estimation method: {center_est_method}")

    if up_est_method == "ground":
        # estimate up direction as the normal of the estimated ground plane
        # use RANSAC to estimate the ground plane in the point cloud
        import pyransac3d as pyrsc

        ground = pyrsc.Plane()
        plane_eq, inliers = ground.fit(
            pts.numpy(), thresh=0.01
        plane_eq = torch.as_tensor(plane_eq)  # A, B, C, D in Ax + By + Cz + D = 0
        z = F.normalize(plane_eq[:3], dim=-1)  # plane normal as up direction
        signed_distance = (torch.cat([pts, torch.ones_like(pts[..., 0:1])], dim=-1) * plane_eq).sum(
            -1
        )
        if signed_distance.mean() < 0:
            z = -z  # flip the direction if points lie under the plane
    elif up_est_method == "camera":
        # estimate up direction as the average of all camera up directions
        z = F.normalize((poses[..., 3] - center).mean(0), dim=0)
    else:
        raise NotImplementedError(f"Unknown up estimation method: {up_est_method}")

    # new axis
    y_ = torch.as_tensor([z[1], -z[0], 0.0])
    x = F.normalize(y_.cross(z), dim=0)
    y = z.cross(x)

    if center_est_method == "point":
        raise NotImplementedError
        # rotation
        Rc = torch.stack([x, y, z], dim=1)
        R = Rc.T
        poses_homo = torch.cat(
            [
                poses,
                torch.as_tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(poses.shape[0], -1, -1),
            ],
            dim=1,
        )
        inv_trans = torch.cat(
            [
                torch.cat([R, torch.as_tensor([[0.0, 0.0, 0.0]]).T], dim=1),
                torch.as_tensor([[0.0, 0.0, 0.0, 1.0]]),
            ],
            dim=0,
        )
        poses_norm = (inv_trans @ poses_homo)[:, :3]
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:, 0:1])], dim=-1)[..., None])[
            :, :3, 0
        ]

        # translation and scaling
        poses_min, poses_max = (
            poses_norm[..., 3].min(0)[0],
            poses_norm[..., 3].max(0)[0],
        )
        pts_fg = pts[
            (poses_min[0] < pts[:, 0])
            & (pts[:, 0] < poses_max[0])
            & (poses_min[1] < pts[:, 1])
            & (pts[:, 1] < poses_max[1])
        ]
        center = get_center(pts_fg)
        tc = center.reshape(3, 1)
        t = -tc
        poses_homo = torch.cat(
            [
                poses_norm,
                torch.as_tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(poses_norm.shape[0], -1, -1),
            ],
            dim=1,
        )
        inv_trans = torch.cat(
            [
                torch.cat([torch.eye(3), t], dim=1),
                torch.as_tensor([[0.0, 0.0, 0.0, 1.0]]),
            ],
            dim=0,
        )
        poses_norm = (inv_trans @ poses_homo)[:, :3]
        scale = poses_norm[..., 3].norm(p=2, dim=-1).min()
        poses_norm[..., 3] /= scale
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:, 0:1])], dim=-1)[..., None])[
            :, :3, 0
        ]
        pts = pts / scale
    else:
        # rotation and translation
        Rc = torch.stack([x, y, z], dim=1)
        tc = center.reshape(3, 1)
        R, t = Rc.T, -Rc.T @ tc
        poses_homo = torch.cat(
            [
                poses,
                torch.as_tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(poses.shape[0], -1, -1),
            ],
            dim=1,
        )
        inv_trans = torch.cat(
            [torch.cat([R, t], dim=1), torch.as_tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0
        )
        poses_norm = (inv_trans @ poses_homo)[:, :3]  # (N_images, 4, 4)

        # scaling
        scale = poses_norm[..., 3].norm(p=2, dim=-1).min()
        poses_norm[..., 3] /= scale

        # apply the transformation to the point cloud
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:, 0:1])], dim=-1)[..., None])[
            :, :3, 0
        ]
        pts = pts / scale

    return poses_norm, pts


def setup_multiscale_filenames(list, n):
    # Validate if length of the list is divisible by n
    if len(list) % n != 0:
        raise ValueError("Length of the list is not divisible by n")

    result = []
    for i in range(0, len(list), n):
        group = list[i : i + n]
        if len(set(group)) != 1:
            raise ValueError(f"Elements from {i} to {i+n-1} are not the same")

        # Only rename the 2nd, 3rd, ... n-th elements
        result.append(group[0])
        for j in range(1, n):
            folder = group[0].parent.with_name(f"{group[0].parent.name}_{2**j}")
            new_path = folder / group[j].name
            result.append(new_path)

            # save image
            if not new_path.exists():
                folder.mkdir(parents=True, exist_ok=True)
                img = cv2.imread(str(group[0]))
                # downsample by factor 2**j, using average pooling
                cv2.imwrite(
                    str(new_path),
                    cv2.resize(
                        img, None, fx=1 / 2**j, fy=1 / 2**j, interpolation=cv2.INTER_LINEAR
                    ),
                )

    return result


def get_human_poses(poses, fixed_camera=False):
    """
    here `poses` are supposed to be w2c !!!!! WTF.
    """
    if poses.ndim == 2:
        assert poses.shape == (3, 4)
        poses = poses[None, ...]

    assert poses.ndim == 3 and poses.shape[1:] == (3, 4)
    pn = poses.shape[0]

    # c2w -> w2c
    poses = torch.cat([poses, torch.tensor([0, 0, 0, 1]).repeat(pn, 1, 1).to(poses)], -2)
    poses = torch.linalg.inv(poses)[:, :3, :]

    cam_cen = (-poses[:, :, :3].permute(0, 2, 1) @ poses[:, :, 3:])[..., 0]  # pn,3
    if fixed_camera:
        pass
    else:
        cam_cen[..., 2] = 0

    Y = torch.zeros([1, 3]).expand(pn, 3).to(poses.device)
    # Y[:, 2] = -1.0
    # why modified: change -1 to 1, as openGL's -z direction is colmap's +z.
    Y[:, 2] = 1.0
    Z = torch.clone(poses[:, 2, :3])  # pn, 3
    Z[:, 2] = 0
    Z = F.normalize(Z, dim=-1)
    X = torch.cross(Y, Z)  # pn, 3
    R = torch.stack([X, Y, Z], 1)  # pn,3,3
    t = -R @ cam_cen[:, :, None]  # pn,3,1
    return torch.cat([R, t], -1)


def _get_image_indices(image_filenames, split, eval_images=None, single_eval=True, eval_only=False):
    num = len(image_filenames)
    all_indices = list(range(num))

    # # if split == "render_mesh":
    # return all_indices

    if eval_images is None:
        # use the first image
        if split == "train":
            res = all_indices[1:]
            if eval_only:
                res = res[:1]
            return res
        else:
            return all_indices[:1]

    elif eval_images[0].isdigit():
        # select eval every k images
        n_eval = int(eval_images[0])
        assert n_eval < num and len(eval_images) == 1
        if n_eval == 0:
            if split == "train":
                return all_indices
            else:
                return []
        step_size = num // n_eval
        all_indices = np.array(all_indices)
        eval_indices = all_indices[::step_size][:n_eval]
        train_indices = np.setdiff1d(all_indices, eval_indices)
        if split == "train":
            res = train_indices.tolist()
            if eval_only:
                res = res[:1]
            return res
        else:
            return eval_indices.tolist() if not single_eval else eval_indices[-1:].tolist()
    else:
        # use predefined eval images
        image_filenames = np.array([x.name for x in image_filenames])
        eval_mask = np.isin(image_filenames, eval_images)
        eval_indices = np.where(eval_mask)[0]
        assert len(eval_indices) > 0
        if split == "train":
            res = np.setdiff1d(all_indices, eval_indices).tolist()
            if eval_only:
                res = res[:1]
            return res
        else:
            return eval_indices.tolist() if not single_eval else eval_indices[-1:].tolist()


@dataclass
class NePColmapDataParserConfig(ColmapDataParserConfig):
    _target: Type = field(default_factory=lambda: NePColmapDataParser)
    """target class to instantiate"""
    points_norm: bool = True
    """use obj point clouds to normalize the world."""
    colmap_coord: bool = False
    """use colmap WORLD system to train the model"""
    multiscale: int = 0
    """n for downscaling times, the minimal size factor 2 ** n"""
    eval_images: Optional[Union[List[str], int]] = None
    """filename or num of the eval images"""
    single_eval: bool = True
    """if true, eval the same image at each eval step"""
    eval_only: bool = False
    use_nsr: bool = False
    up_from_pose: bool = True
    sanity: bool = False


ds_meta_info = {
    "bear": {
        "forward": np.asarray([0.539944, -0.342791, 0.341446], np.float32),
        "up": np.asarray((0.0512875, -0.645326, -0.762183), np.float32),
        "fixed_camera": False,
    },
    "coral": {
        "forward": np.asarray([0.004226, -0.235523, 0.267582], np.float32),
        "up": np.asarray((0.0477973, -0.748313, -0.661622), np.float32),
        "fixed_camera": False,
    },
    "maneki": {
        "forward": np.asarray([-2.336584, -0.406351, 0.482029], np.float32),
        "up": np.asarray((-0.0117387, -0.738751, -0.673876), np.float32),
        "fixed_camera": True,
    },
    "bunny": {
        "forward": np.asarray([0.437076, -1.672467, 1.436961], np.float32),
        "up": np.asarray((-0.0693234, -0.644819, -0.761185), np.float32),
        "fixed_camera": True,
    },
    "vase": {
        "forward": np.asarray([-0.792947, -0.099202, 0.142175], np.float32),
        "up": np.asarray((-0.0255325, -0.740914, -0.671114), np.float32),
    },
    "dog": {
        "forward": np.asarray([0.088698, 0.011980, 0.030138], np.float32),
        "up": np.asarray((0.121074, -0.388078, -0.913639), np.float32),
    },
    "dog2": {
        "forward": np.asarray([0.196224, -0.888263, 1.019088], np.float32),
        "up": np.asarray((-0.0817993, -0.759058, -0.645864), np.float32),
    },
    "dog3": {
        "forward": np.asarray([-0.813728, 0.622674, -0.223115], np.float32),
        "up": np.asarray((0.093104, -0.225795, -0.969716), np.float32),
    },
    "rabbit": {
        "forward": np.asarray([0.215735, -0.887614, 1.811427], np.float32),
        "up": np.asarray((-0.149655, -0.894801, -0.420636), np.float32),
    },
    "boat": {
        "forward": np.asarray([-0.225158, 0.070307, -0.278951], np.float32),
        "up": np.asarray((0.105032, -0.94092, -0.321929), np.float32),
    },
    # our syn data
    "bunny_d": {
        "forward": np.asarray([-0.416398, 0.300811, -1.054823], np.float32),
        "up": np.asarray((0.034747, -0.960158, -0.277134), np.float32),
    },
    "dog_m2m": {
        "forward": np.asarray([1.186295, 0.631453, -0.679657], np.float32),
        "up": np.asarray((-0.0509323, -0.685772, -0.726032), np.float32),
    },
}
for name in ["armadillo_d", "head_d", "box_d"]:
    ds_meta_info[name] = ds_meta_info["bunny_d"]


@dataclass
class NePColmapDataParser(ColmapDataParser):
    config: NePColmapDataParserConfig

    def __init__(self, config: NePColmapDataParserConfig):
        """
        meta_info: forward is x direction indicated by user in cloudcompare;
                    up is negtive z direction indicated by user in cloudcompare - in opencv coordinate system.
        """
        super().__init__(config)
        self.meta_info = ds_meta_info
        assert not self.config.use_nsr
        if not self.config.use_nsr:
            self._setup_rescale_recenter_mtx()
        # self._downscale_factor = self.config.downscale_factor

    @staticmethod
    def _load_point_cloud(pcl_path):
        with open(pcl_path, "rb") as f:
            plydata = plyfile.PlyData.read(f)
            xyz = np.stack(
                [np.array(plydata["vertex"][c]).astype(float) for c in ("x", "y", "z")],
                axis=1,
            )
        return xyz

    def _setup_rescale_recenter_mtx(self):
        if not self.config.points_norm:
            return

        ref_points = self._load_point_cloud(self.config.data / "object_point_cloud.ply")
        max_pt, min_pt = np.max(ref_points, 0), np.min(ref_points, 0)
        center = (max_pt + min_pt) * 0.5
        self.colmap_world_center = center
        self.colmap_world_scale = np.max(np.linalg.norm(ref_points - center[None, :], 2, 1))

        scene_name = self.config.data.stem
        if not self.config.up_from_pose:
            assert scene_name in self.meta_info
            up, forward = (
                self.meta_info[scene_name]["up"],
                self.meta_info[scene_name]["forward"],
            )
            # self.fixed_camera = self.meta_info[scene_name]["fixed_camera"]
            print(f"Reading x, z direction from self.meta_info: {forward}, {up}")
            self.colmap_world_up = up / np.linalg.norm(up)

    def orient_and_center_poses(
        self, poses: Float[Tensor, "*num_poses 4 4"]
    ) -> Tuple[Float[Tensor, "*num_poses 3 4"], Float[Tensor, "3 4"]]:
        """
        copied from camera_utils.auto_orient_and_center_poses
        """
        # convert colmap xyz to opengl xyz: switch zy & invert z
        translation = torch.tensor(self.colmap_world_center, dtype=torch.float32)
        # assert self.config.colmap_coord == False
        if not self.config.colmap_coord:
            print("Train model in ngp world system")
            translation = translation[[1, 0, 2]]
            translation[-1] *= -1
            print("OpenGL transform:", translation)
        else:
            print("Train model in colmap world system")

        if self.config.up_from_pose:
            up = torch.mean(poses[:, :3, 1], dim=0)
            up = up / torch.linalg.norm(up)

        else:
            up = torch.tensor(self.colmap_world_up, dtype=torch.float32)
            if not self.config.colmap_coord:
                up = up[[1, 0, 2]]
                up[-1] *= -1

        rotation = rotation_matrix(up, torch.Tensor([0, 0, 1]))
        transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses
        return oriented_poses, transform

    def _get_image_indices(self, image_filenames, split):
        return _get_image_indices(
            image_filenames,
            split,
            self.config.eval_images,
            self.config.single_eval,
            self.config.eval_only,
        )

    def _generate_dataparser_outputs(self, split: str = "train"):
        """
        copied from ColmapDataParser._generate_dataparser_outputs. the only modifications:
            - use self.colmap_world_up and self.colmap_world_center to recenter and orient the world
            - use self.scale_from_points to rescale the world
        """
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        colmap_path = self.config.data / self.config.colmap_path
        assert colmap_path.exists(), f"Colmap path {colmap_path} does not exist."

        meta = self._get_all_images_and_cameras(colmap_path)
        camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in meta["frames"]:
            fx.append(float(frame["fl_x"]))
            fy.append(float(frame["fl_y"]))
            cx.append(float(frame["cx"]))
            cy.append(float(frame["cy"]))
            height.append(int(frame["h"]))
            width.append(int(frame["w"]))

            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )

            image_filenames.append(Path(frame["file_path"]))
            poses.append(frame["transform_matrix"])
            if "mask_path" in frame:
                mask_filenames.append(Path(frame["mask_path"]))
            if "depth_path" in frame:
                depth_filenames.append(Path(frame["depth_path"]))

            # create multiscale dataset
            if split == "train" and self.config.multiscale > 0:
                for j in range(self.config.multiscale):
                    factor = 2 ** (j + 1)
                    fx.append(float(frame["fl_x"]) / factor)
                    fy.append(float(frame["fl_y"]) / factor)
                    cx.append(float(frame["cx"]) / factor)
                    cy.append(float(frame["cy"]) / factor)
                    height.append(int(frame["h"]) / factor)
                    width.append(int(frame["w"]) / factor)

                    # others
                    distort.append(distort[-1])
                    image_filenames.append(image_filenames[-1])
                    poses.append(poses[-1])
                    assert "mask_path" not in frame and "depth_path" not in frame

        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        # why modified: use center and up from ref_points to recenter and orient the world
        # here poses are in opengl coords
        poses = torch.from_numpy(np.array(poses).astype(np.float32))

        if self.config.use_nsr:
            print("Use NSR to normalize the world.")
            pts3d = read_points3d_binary(self.config.data / "sparse/0/points3D.bin")
            pts3d = torch.from_numpy(np.array([pts3d[k].xyz for k in pts3d])).float()
            poses, pts3d = normalize_poses(
                poses,
                pts3d,
                up_est_method="camera",
                center_est_method="lookat",
            )
            transform_matrix = None

        elif self.config.points_norm:
            print("Rescale-recenter the world based on the obj point clouds.")
            poses, transform_matrix = self.orient_and_center_poses(poses)

        else:
            print("Rescale-recenter the world based on camera poses.")
            poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
                poses,
                method=self.config.orientation_method,
                center_method=self.config.center_method,
            )

        # why modified: use the ref_points to scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses and not self.config.use_nsr:
            scale_from_poses = float(torch.max(torch.abs(poses[:, :3, 3])))
            if self.config.points_norm:
                scale_factor /= self.colmap_world_scale
                print(
                    f"World scale from points: {1 / self.colmap_world_scale:.3f}; from poses: {1 / scale_from_poses:.3f}"
                )
            else:
                scale_factor /= scale_from_poses
        # scale_factor *= self.config.scale_factor
        print(f"Final world scale factor: {scale_factor:.3f}")

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        # self._downscale_factor = self.config.downscale_factor
        indices = self._get_image_indices(image_filenames, split)
        (
            image_filenames,
            mask_filenames,
            depth_filenames,
            downscale_factor,
        ) = self._setup_downscale_factor(image_filenames, mask_filenames, depth_filenames)

        if image_filenames[0].exists():
            image_filenames = [image_filenames[i] for i in indices]
        else:
            image_filenames = [self.config.data / image_filenames[i] for i in indices]
            assert image_filenames[0].exists(), f"{image_filenames[0]} does not exist."

        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-aabb_scale, -aabb_scale, -aabb_scale],
                    [aabb_scale, aabb_scale, aabb_scale],
                ],
                dtype=torch.float32,
            )
        )

        fx = torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            # metadata={
            #     # why modified:
            #     'human_poses': get_human_poses(poses[:, :3, :4], self.fixed_camera).view(-1, 12)
            # }
        )
        cameras.rescale_output_resolution(scaling_factor=1.0 / downscale_factor)

        if "applied_transform" in meta and transform_matrix is not None:
            applied_transform = torch.tensor(
                meta["applied_transform"], dtype=transform_matrix.dtype
            )
            transform_matrix = transform_matrix @ torch.cat(
                [
                    applied_transform,
                    torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype),
                ],
                0,
            )
            transform_args = dict(dataparser_transform=transform_matrix)
        else:
            transform_args = dict()

        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        # resize images if multiscale
        if split == "train" and self.config.multiscale > 0:
            image_filenames = setup_multiscale_filenames(
                image_filenames, self.config.multiscale + 1
            )

        if not self.config.sanity:
            return DataparserOutputs(
                image_filenames=image_filenames,
                cameras=cameras,
                scene_box=scene_box,
                mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
                dataparser_scale=scale_factor,
                metadata={
                    "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                    "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                },
                **transform_args,
            )
        else:
            sanity_num = 1
            return DataparserOutputs(
                image_filenames=image_filenames[:sanity_num],
                cameras=cameras[:sanity_num],
                scene_box=scene_box,
                mask_filenames=mask_filenames[:sanity_num] if len(mask_filenames) > 0 else None,
                dataparser_scale=scale_factor,
                metadata={
                    "depth_filenames": depth_filenames[sanity_num]
                    if len(depth_filenames) > 0
                    else None,
                    "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                },
                **transform_args,
            )


@dataclass
class NeroSynDataParserConfig(BlenderDataParserConfig):
    _target: Type = field(default_factory=lambda: NeroSynDataParser)
    eval_images: Optional[List[str]] = None
    aabb_min: List[float] = field(default_factory=lambda: [-1, -1, -1])
    aabb_max: List[float] = field(default_factory=lambda: [1, 1, 1])
    """filename of the eval images"""


@dataclass
class NeroSynDataParser(Blender):
    config: NeroSynDataParserConfig

    def __init__(self, config: NeroSynDataParserConfig):
        super().__init__(config)

        # get world center and scale
        print("AABB min:", self.config.aabb_min, "AABB max:", self.config.aabb_max)
        self.aabb = torch.tensor([self.config.aabb_min, self.config.aabb_max])
        self.world_center = self.aabb.mean(0)
        self.scale_factor /= (self.aabb[1] - self.world_center).max()
        print("World center:", self.world_center, "World scale:", self.scale_factor)

    def recenter_poses(
        self, poses: Float[Tensor, "*num_poses 4 4"]
    ) -> Tuple[Float[Tensor, "*num_poses 3 4"], Float[Tensor, "3 4"]]:
        poses = torch.tensor(poses)
        translation = torch.tensor(self.world_center, dtype=torch.float32)
        rotation = torch.eye(3)
        transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
        res_poses = transform @ poses
        return torch.cat([res_poses, torch.tensor([[0, 0, 0, 1]])], dim=0), transform

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        meta = load_from_json(self.data / f"transforms.json")
        image_filenames = []
        poses = []

        eval_fn = lambda x: Path(x["file_path"] + ".png").name in self.config.eval_images

        train_fn = lambda x: not eval_fn(x)
        filter_fn = train_fn if split == "train" else eval_fn

        for frame in filter(filter_fn, meta["frames"]):
            fname = self.data / Path(frame["file_path"] + ".png").name
            image_filenames.append(fname)
            pose, _ = self.recenter_poses(frame["transform_matrix"])
            poses.append(pose.numpy())
        poses = np.array(poses).astype(np.float32)

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # note: div by sqrt(2) is necessary when using sphere bounding
        camera_to_world[..., 3] *= self.scale_factor / math.sqrt(2)
        scene_box = SceneBox(aabb=torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32))

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs


nep_real_data = DataParserSpecification(config=NePColmapDataParserConfig())
nep_syn_data = DataParserSpecification(config=NeroSynDataParserConfig())
