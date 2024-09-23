import abc
import glob
import os
import random
from pathlib import Path

import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm

from nep.colmap import plyfile
from nep.colmap.read_write_model import read_model
from nep.utils.base_utils import resize_img, read_pickle, project_points, save_pickle, pose_inverse, \
    mask_depth_to_pts, pose_apply
import open3d as o3d

from nep.utils.pose_utils import look_at_crop


class BaseDatabase(abc.ABC):
    def __init__(self, database_name, cfg):
        self.database_name = database_name
        self.cfg = cfg

    @abc.abstractmethod
    def get_image(self, img_id):
        pass

    @abc.abstractmethod
    def get_K(self, img_id):
        pass

    @abc.abstractmethod
    def get_pose(self, img_id):  # gt poses
        pass

    @abc.abstractmethod
    def get_img_ids(self):
        pass

    @abc.abstractmethod
    def get_depth(self, img_id):
        pass


def crop_by_points(img, ref_points, pose, K, size):
    h, w, _ = img.shape
    pts2d, depth = project_points(ref_points, pose, K)
    pts2d[:, 0] = np.clip(pts2d[:, 0], a_min=0, a_max=w - 1)
    pts2d[:, 1] = np.clip(pts2d[:, 1], a_min=0, a_max=h - 1)
    pt_min, pt_max = np.min(pts2d, 0), np.max(pts2d, 0)

    region_size = np.max(pt_max - pt_min)
    region_size = min(region_size, h - 3, w - 3)  # cannot exceeds image size

    x_size, y_size = pt_max - pt_min
    x_min, y_min = pt_min
    x_max, y_max = pt_max
    if region_size <= x_size:
        x_cen = (x_min + x_max) / 2
    elif region_size > x_size:
        b0 = max(region_size / 2, x_max - region_size / 2)
        b1 = min(x_min + region_size / 2, w - 2 - region_size / 2)
        x_cen = (b0 + b1) / 2
    if region_size <= y_size:
        y_cen = (y_min + y_max) / 2
    elif region_size > y_size:
        b0 = max(region_size / 2, y_max - region_size / 2)
        b1 = min(y_min + region_size / 2, h - 2 - region_size / 2)
        y_cen = (b0 + b1) / 2

    center = np.asarray([x_cen, y_cen], np.float32)
    scale = size / region_size
    img1, K1, pose1, pose_rect, H = look_at_crop(
        img, K, pose, center, 0, scale, size, size)
    return img1, K1, pose1


class GlossyRealDatabase(BaseDatabase):
    """
    meta_info: +x and +z from cloudcompare app.
    """
    meta_info = {
        'bear': {'forward': np.asarray([0.539944, -0.342791, 0.341446], np.float32), 'up': np.asarray((0.0512875, -0.645326, -0.762183), np.float32), },
        'coral': {'forward': np.asarray([0.004226, -0.235523, 0.267582], np.float32), 'up': np.asarray((0.0477973, -0.748313, -0.661622), np.float32), },
        'maneki': {'forward': np.asarray([-2.336584, -0.406351, 0.482029], np.float32), 'up': np.asarray((-0.0117387, -0.738751, -0.673876), np.float32), },
        'bunny': {'forward': np.asarray([0.437076, -1.672467, 1.436961], np.float32), 'up': np.asarray((-0.0693234, -0.644819, -.761185), np.float32), },
        'vase': {'forward': np.asarray([-0.792947, -0.099202, 0.142175], np.float32), 'up': np.asarray((-0.0255325, -0.740914, -0.671114), np.float32), },
        'dog': {'forward': np.asarray([0.088698, 0.011980, 0.030138], np.float32), 'up': np.asarray((0.121074, -0.388078, -0.913639), np.float32), },
        'dog2': {'forward': np.asarray([0.196224, -0.888263, 1.019088], np.float32), 'up': np.asarray((-0.0817993, -0.759058, -0.645864), np.float32), },
        'dog3': {'forward': np.asarray([-0.813728, 0.622674, -0.223115], np.float32), 'up': np.asarray((0.093104, -0.225795, -0.969716), np.float32), },
    }

    def __init__(self, database_name, cfg, root=None):
        super().__init__(database_name, cfg)
        _, self.object_name, self.max_len = database_name.split('/')

        if root is not None:
            self.root = root
        else:
            self.root = f'data/GlossyReal/{self.object_name}'

        self.opengl_coords = cfg['opengl_coords']
        if self.opengl_coords:
            print('Use opengl_coords instead of opencv.')

        self._parse_colmap()
        self._normalize()
        if not self.max_len.startswith('raw'):
            self.max_len = int(self.max_len)
            self.image_dir = ''
            self._crop()
        else:
            h, w, _ = imread(
                f'{self.root}/images/{self.image_names[self.img_ids[0]]}').shape
            max_len = int(self.max_len.split('_')[1])
            ratio = float(max_len) / max(h, w)
            th, tw = int(ratio*h), int(ratio*w)
            rh, rw = th / h, tw / w
            self.h, self.w = th, tw

            Path(
                f'{self.root}/images_{self.max_len}').mkdir(exist_ok=True, parents=True)
            for img_id in tqdm(self.img_ids):
                if not Path(f'{self.root}/images_{self.max_len}/{self.image_names[img_id]}').exists():
                    img = imread(
                        f'{self.root}/images/{self.image_names[img_id]}')
                    img = resize_img(img, ratio)
                    imsave(
                        f'{self.root}/images_{self.max_len}/{self.image_names[img_id]}', img)

                K = self.Ks[img_id]
                self.Ks[img_id] = np.diag([rw, rh, 1.0]) @ K

    def _parse_colmap(self):
        if Path(f'{self.root}/cache.pkl').exists():
            self.poses, self.Ks, self.image_names, self.img_ids, self.camera_model = read_pickle(
                f'{self.root}/cache.pkl')
        else:
            cameras, images, points3d = read_model(
                f'{self.root}/colmap/sparse/0')

            self.poses, self.Ks, self.image_names, self.img_ids = {}, {}, {}, []
            self.camera_model = None
            for img_id, image in images.items():
                if not Path(f'{self.root}/images/{image.name}').exists():
                    print(
                        f'Warning: image {image.name} is in colmap database, but can not be found in image folder.')
                    continue

                self.img_ids.append(img_id)
                self.image_names[img_id] = image.name

                R = image.qvec2rotmat()
                t = image.tvec
                pose = np.concatenate([R, t[:, None]], 1).astype(np.float32)

                # opencv -> opengl
                # if self.opengl_coords:
                #     pose[0:3, 1:3] *= -1
                #     pose = pose[np.array([1, 0, 2]), :]
                #     pose[2, :] *= -1

                self.poses[img_id] = pose

                cam_id = image.camera_id
                camera = cameras[cam_id]

                # save camera_model
                if self.camera_model is None:
                    self.camera_model = camera.model
                else:
                    assert self.camera_model == camera.model

                if camera.model == 'SIMPLE_RADIAL':
                    f, cx, cy, _ = camera.params
                elif camera.model == 'SIMPLE_PINHOLE':
                    f, cx, cy = camera.params
                else:
                    raise NotImplementedError
                self.Ks[img_id] = np.asarray(
                    [[f, 0, cx], [0, f, cy], [0, 0, 1], ], np.float32)

            save_pickle([self.poses, self.Ks, self.image_names,
                        self.img_ids, self.camera_model], f'{self.root}/cache.pkl')

    def _load_point_cloud(self, pcl_path):
        with open(pcl_path, "rb") as f:
            plydata = plyfile.PlyData.read(f)
            xyz = np.stack([np.array(plydata["vertex"][c]).astype(float)
                           for c in ("x", "y", "z")], axis=1)
        return xyz

    def _compute_rotation(self, vert, forward):
        y = np.cross(vert, forward)
        x = np.cross(y, vert)

        vert = vert/np.linalg.norm(vert)
        x = x/np.linalg.norm(x)
        y = y/np.linalg.norm(y)
        R = np.stack([x, y, vert], 0)
        return R

    def _normalize(self):
        ref_points = self._load_point_cloud(
            f'{self.root}/object_point_cloud.ply')
        max_pt, min_pt = np.max(ref_points, 0), np.min(ref_points, 0)
        center = (max_pt + min_pt) * 0.5
        offset = -center  # x1 = x0 + offset
        # x2 = scale * x1
        scale = 1 / np.max(np.linalg.norm(ref_points - center[None, :], 2, 1))
        if self.object_name in self.meta_info:
            up, forward = self.meta_info[self.object_name]['up'], self.meta_info[self.object_name]['forward']
            print(f'Reading x, z direction from self.meta_info: {forward}, {up}')
        else:
            up, forward = np.loadtxt(f'{self.root}/meta_info.txt')
            print(f'Reading x, z direction from meta_info.txt: {forward}, {up}')

        up, forward = up/np.linalg.norm(up), forward/np.linalg.norm(forward)
        R_rec = self._compute_rotation(up, forward)  # x3 = R_rec @ x2
        self.ref_points = scale * (ref_points + offset) @ R_rec.T
        self.max_pt, self.min_pt = np.max(self.ref_points, 0), np.min(self.ref_points, 0)
        self.scale_rect = scale
        self.offset_rect = offset
        self.R_rect = R_rec

        # x3 = R_rec @ (scale * (x0 + offset))
        # R_rec.T @ x3 / scale - offset = x0

        # pose [R,t] x_c = R @ x0 + t
        # pose [R,t] x_c = R @ (R_rec.T @ x3 / scale - offset) + t
        # x_c = R @ R_rec.T @ x3 + (t - R @ offset) * scale
        # R_new = R @ R_rec.T    t_new = (t - R @ offset) * scale
        for img_id, pose in self.poses.items():
            R, t = pose[:, :3], pose[:, 3]
            R_new = R @ R_rec.T
            t_new = (t - R @ offset) * scale
            self.poses[img_id] = np.concatenate([R_new, t_new[:, None]], -1)

    def _crop(self):
        if Path(f'{self.root}/images_{self.max_len}/meta_info.pkl').exists():
            self.poses, self.Ks = read_pickle(
                f'{self.root}/images_{self.max_len}/meta_info.pkl')
        else:
            poses_new, Ks_new = {}, {}
            print('cropping images ...')
            Path(
                f'{self.root}/images_{self.max_len}').mkdir(exist_ok=True, parents=True)
            for img_id in tqdm(self.img_ids):
                pose, K = self.poses[img_id], self.Ks[img_id]
                img = imread(f'{self.root}/images/{self.image_names[img_id]}')
                img1, K1, pose1 = crop_by_points(
                    img, self.ref_points, pose, K, self.max_len)
                imsave(
                    f'{self.root}/images_{self.max_len}/{self.image_names[img_id]}', img1)
                poses_new[img_id] = pose1
                Ks_new[img_id] = K1

            save_pickle([poses_new, Ks_new],
                        f'{self.root}/images_{self.max_len}/meta_info.pkl')
            self.poses, self.Ks = poses_new, Ks_new

    def get_image(self, img_id):
        img = imread(self.get_image_path(img_id))
        return img

    def get_image_path(self, img_id):
        return f'{self.root}/images_{self.max_len}/{self.image_names[img_id]}' 

    def get_K(self, img_id):
        K = self.Ks[img_id]
        return K.copy()

    def get_pose(self, img_id):
        return self.poses[img_id].copy()

    def get_img_ids(self):
        return self.img_ids

    def get_mask(self, img_id):
        raise NotImplementedError

    def get_depth(self, img_id):
        img = self.get_image(img_id)
        h, w, _ = img.shape
        return np.ones([h, w], np.float32), np.ones([h, w], np.bool_)


class GlossySyntheticDatabase(BaseDatabase):
    # todo: add aabb fro syn data
    def __init__(self, database_name, cfg):
        super().__init__(database_name, cfg)
        _, model_name = database_name.split('/')
        RENDER_ROOT = 'data/GlossySynthetic'
        self.root = f'{RENDER_ROOT}/{model_name}'
        self.img_num = len(glob.glob(f'{self.root}/*.pkl'))
        self.img_ids = [str(k) for k in range(self.img_num)]
        self.cams = [read_pickle(f'{self.root}/{k}-camera.pkl')
                     for k in range(self.img_num)]
        self.scale_factor = 1.0
        # self.opengl_coords = cfg['opengl_coords']
        # if self.opengl_coords:
        #     print('Use opengl_coords instead of opencv.')

    def get_image(self, img_id):
        return imread(self.get_image_path(img_id))[..., :3]

    def get_image_path(self, img_id):
        return f'{self.root}/{img_id}.png'

    def get_K(self, img_id):
        K = self.cams[int(img_id)][1]
        return K.astype(np.float32)

    def get_pose(self, img_id):
        pose = self.cams[int(img_id)][0].copy()
        pose = pose.astype(np.float32)
        pose[:, 3:] *= self.scale_factor
        return pose

    def get_img_ids(self):
        return self.img_ids

    def get_depth(self, img_id):
        assert (self.scale_factor == 1.0)
        depth = imread(f'{self.root}/{img_id}-depth.png')
        depth = depth.astype(np.float32) / 65535 * 15
        mask = depth < 14.5
        return depth, mask

    def get_mask(self, img_id):
        raise NotImplementedError


def parse_database_name(cfg) -> BaseDatabase:
    name2database = {
        'syn': GlossySyntheticDatabase,
        'real': GlossyRealDatabase,
    }
    database_name = cfg['database_name']
    database_type = database_name.split('/')[0]
    if database_type in name2database:
        return name2database[database_type](database_name, cfg)
    else:
        raise NotImplementedError


def get_database_split(database: BaseDatabase, split_type='validation', valid_num=1):
    if split_type == 'validation':
        random.seed(6033)
        img_ids = database.get_img_ids()
        random.shuffle(img_ids)

        # option 1: pick first sample for test
        if valid_num == 1:
            test_ids = img_ids[:1]
            train_ids = img_ids[1:]
        else:
            # option 2: ramdom pick 4 samples for test. for debug only.
            n = valid_num
            first_img = img_ids.pop(0)
            test_ids = np.random.choice(img_ids, n, replace=False)
            train_ids = np.setdiff1d(img_ids, test_ids).tolist()
            test_ids = test_ids.tolist()
            test_ids.append(first_img)

    elif split_type == 'test':
        database_name = database.database_name
        assert database_name.startswith('render')
        test_ids, train_ids = read_pickle('configs/synthetic_split_128.pkl')
    else:
        raise NotImplementedError
    return train_ids, test_ids


def get_database_eval_points(database):
    if isinstance(database, GlossySyntheticDatabase):
        fn = f'{database.root}/eval_pts.ply'
        if os.path.exists(fn):
            pcd = o3d.io.read_point_cloud(str(fn))
            return np.asarray(pcd.points)
        _,  test_ids = get_database_split(database, 'test')
        pts = []
        for img_id in test_ids:
            depth, mask = database.get_depth(img_id)
            K = database.get_K(img_id)
            pts_ = mask_depth_to_pts(mask, depth, K)
            pose = pose_inverse(database.get_pose(img_id))
            pts_ = pose_apply(pose, pts_)
            pts.append(pts_)
        pts = np.concatenate(pts, 0).astype(np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        downpcd = pcd.voxel_down_sample(voxel_size=0.01)
        o3d.io.write_point_cloud(fn, downpcd)
        print(f'point number {len(downpcd.points)} ...')
        return np.asarray(downpcd.points, np.float32)
    else:
        raise NotImplementedError
