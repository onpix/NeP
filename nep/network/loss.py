import numpy as np
import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def orientation_loss(
    weights: Float[Tensor, "*bs num_samples 1"],
    normals: Float[Tensor, "*bs num_samples 3"],
    viewdirs: Float[Tensor, "*bs 3"],
):
    """Orientation loss proposed in Ref-NeRF.
    Loss that encourages that all visible normals are facing towards the camera.
    """
    w = weights
    n = normals
    v = viewdirs * -1
    n_dot_v = (n * v[..., None, :]).sum(dim=-1)
    return (w[..., 0] * torch.fmin(torch.zeros_like(n_dot_v), n_dot_v) ** 2).sum(dim=-1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, stride=None):
    mu1 = F.conv2d(img1, window, padding=(window_size - 1) // 2, groups=channel, stride=stride)
    mu2 = F.conv2d(img2, window, padding=(window_size - 1) // 2, groups=channel, stride=stride)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=(window_size - 1) // 2, groups=channel, stride=stride)
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=(window_size - 1) // 2, groups=channel, stride=stride)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=(window_size - 1) // 2, groups=channel, stride=stride)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=3, size_average=True, stride=3):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.stride = stride
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """
        img1, img2: torch.Tensor([b,c,h,w])
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(
            img1, img2, window, self.window_size, channel, self.size_average, stride=self.stride
        )


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def closest_factors(N):
    min_diff = float("inf")
    result = (1, N)

    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            j = N // i
            if abs(i - j) < min_diff:
                min_diff = abs(i - j)
                result = (i, j)

    return result


class S3IM(torch.nn.Module):
    r"""Implements Stochastic Structural SIMilarity(S3IM) algorithm.
    It is proposed in the ICCV2023 paper
    `S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields`.

    Arguments:
        kernel_size (int): kernel size in ssim's convolution(default: 4)
        stride (int): stride in ssim's convolution(default: 4)
        repeat_time (int): repeat time in re-shuffle virtual patch(default: 10)
        patch_height (height): height of virtual patch(default: 64)
        patch_width (height): width of virtual patch(default: 64)
    """

    def __init__(self, config, kernel_size=4, stride=4, repeat_time=10):
        super(S3IM, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.repeat_time = repeat_time
        self.patch_height, self.patch_width = None, None
        self.ssim_loss = SSIM(window_size=self.kernel_size, stride=self.stride)

    def forward(self, data, batch, *args, **kwargs):
        # def forward(self, src_vec, tar_vec):
        src_vec, tar_vec = data["ray_rgb"], batch["image"]
        if self.patch_height is None or self.patch_height * self.patch_width != src_vec.shape[0]:
            self.patch_height, self.patch_width = closest_factors(src_vec.shape[0])

        index_list = []
        for i in range(self.repeat_time):
            if i == 0:
                tmp_index = torch.arange(len(tar_vec))
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(len(tar_vec))
                index_list.append(ran_idx)
        res_index = torch.cat(index_list)
        tar_all = tar_vec[res_index]
        src_all = src_vec[res_index]
        tar_patch = tar_all.permute(1, 0).reshape(
            1, 3, self.patch_height, self.patch_width * self.repeat_time
        )
        src_patch = src_all.permute(1, 0).reshape(
            1, 3, self.patch_height, self.patch_width * self.repeat_time
        )
        return {"loss_s3im": 1 - self.ssim_loss(src_patch, tar_patch)}


def pred_normal_loss(
    weights: Float[Tensor, "*bs num_samples 1"],
    normals: Float[Tensor, "*bs num_samples 3"],
    pred_normals: Float[Tensor, "*bs num_samples 3"],
):
    """Loss between normals calculated from density and normals from prediction network."""
    return (weights[..., 0] * (1.0 - torch.sum(normals * pred_normals, dim=-1))).sum(dim=-1)


class Loss:
    def __call__(self, data_pr, data_gt, step, **kwargs):
        pass


class NeRFRenderLoss(Loss):
    def __init__(self, cfg):
        pass

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}
        if "loss_rgb" in data_pr:
            outputs["loss_rgb"] = data_pr["loss_rgb"]
        # if 'loss_rgb_fine' in data_pr: outputs['loss_rgb_fine']=data_pr['loss_rgb_fine']
        if "loss_rgb_coarse" in data_pr:
            outputs["loss_rgb_coarse"] = data_pr["loss_rgb_coarse"]
        if "loss_global_rgb" in data_pr:
            outputs["loss_global_rgb"] = data_pr["loss_global_rgb"]
        if "loss_rgb_inner" in data_pr:
            outputs["loss_rgb_inner"] = data_pr["loss_rgb_inner"]
        if "loss_rgb0" in data_pr:
            outputs["loss_rgb0"] = data_pr["loss_rgb0"]
        if "loss_rgb1" in data_pr:
            outputs["loss_rgb1"] = data_pr["loss_rgb1"]
        if "loss_masks" in data_pr:
            outputs["loss_masks"] = data_pr["loss_masks"]
        return outputs


class EikonalLoss(Loss):
    default_cfg = {
        "eikonal_weight": 0.1,
        "eikonal_weight_anneal_begin": 0,
        "eikonal_weight_anneal_end": 0,
    }

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

    def get_eikonal_weight(self, step):
        if step < self.cfg["eikonal_weight_anneal_begin"]:
            return 0.0
        elif (
            self.cfg["eikonal_weight_anneal_begin"] <= step < self.cfg["eikonal_weight_anneal_end"]
        ):
            return (
                self.cfg["eikonal_weight"]
                * (step - self.cfg["eikonal_weight_anneal_begin"])
                / (self.cfg["eikonal_weight_anneal_end"] - self.cfg["eikonal_weight_anneal_begin"])
            )
        else:
            return self.cfg["eikonal_weight"]

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        weight = self.get_eikonal_weight(step)
        outputs = {"loss_eikonal": data_pr["gradient_error"] * weight}
        return outputs


class MaterialRegLoss(Loss):
    default_cfg = {}

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}
        if "loss_mat_reg" in data_pr:
            outputs["loss_mat_reg"] = data_pr["loss_mat_reg"]
        if "loss_diffuse_light" in data_pr:
            outputs["loss_diffuse_light"] = data_pr["loss_diffuse_light"]
        return outputs


class StdRecorder(Loss):
    default_cfg = {
        "apply_std_loss": False,
        "std_loss_weight": 0.05,
        "std_loss_weight_type": "constant",
    }

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}
        if "std" in data_pr:
            outputs["std"] = data_pr["std"]
            if self.cfg["apply_std_loss"]:
                if self.cfg["std_loss_weight_type"] == "constant":
                    outputs["loss_std"] = data_pr["std"] * self.cfg["std_loss_weight"]
                else:
                    raise NotImplementedError
        if "inner_std" in data_pr:
            outputs["inner_std"] = data_pr["inner_std"]
        if "outer_std" in data_pr:
            outputs["outer_std"] = data_pr["outer_std"]
        return outputs


class OccLoss(Loss):
    default_cfg = {}

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}
        if "loss_occ" in data_pr:
            outputs["loss_occ"] = torch.mean(data_pr["loss_occ"]).reshape(1)
        return outputs


class InitSDFRegLoss(Loss):
    def __init__(self, cfg):
        pass

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        reg_step = 1000
        small_threshold = 0.1
        large_threshold = 1.05
        if "sdf_vals" in data_pr and "sdf_pts" in data_pr and step < reg_step:
            norm = torch.norm(data_pr["sdf_pts"], dim=-1)
            sdf = data_pr["sdf_vals"]
            small_mask = norm < small_threshold
            if torch.sum(small_mask) > 0:
                bounds = norm[small_mask] - small_threshold  # 0-small_threshold -> 0
                # we want sdf - bounds < 0
                small_loss = torch.mean(torch.clamp(sdf[small_mask] - bounds, min=0.0))
                small_loss = torch.sum(small_loss) / (torch.sum(small_loss > 1e-5) + 1e-3)
            else:
                small_loss = torch.zeros(1)

            large_mask = norm > large_threshold
            if torch.sum(large_mask) > 0:
                # 0 -> 1 - large_threshold
                bounds = norm[large_mask] - large_threshold
                # we want sdf - bounds > 0 => bounds - sdf < 0
                large_loss = torch.clamp(bounds - sdf[large_mask], min=0.0)
                large_loss = torch.sum(large_loss) / (torch.sum(large_loss > 1e-5) + 1e-3)
            else:
                large_loss = torch.zeros(1)

            anneal_weights = (np.cos((step / reg_step) * np.pi) + 1) / 2
            return {
                "loss_sdf_large": large_loss * anneal_weights,
                "loss_sdf_small": small_loss * anneal_weights,
            }
        else:
            return {}


name2loss = {
    "nerf_render": NeRFRenderLoss,
    "eikonal": EikonalLoss,
    "std": StdRecorder,
    "init_sdf_reg": InitSDFRegLoss,
    # 'occ': OccLoss,
    "mat_reg": MaterialRegLoss,
    "s3im": S3IM,
}
