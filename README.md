# NeP
[CVPR2024] Inverse Rendering of Glossy Objects via the Neural Plenoptic Function and Radiance Fields

## Set up env

1. Install pkgs: `nerfstudio==0.3.4 pytorch>=2.0 open3d pymcubes plyfile`
2. Install the repo as a python package: `pip install -e .`
3. Run the code using `ns-train`

## Download data

[Onedrive Link](https://portland-my.sharepoint.com/:u:/g/personal/hywang26-c_my_cityu_edu_hk/EdvKn7xd829BihTzRvR4alMB7LAxItpE8hJ6z57TEnGt1g?e=aXQ54u)

## Train

```
# geometry learning (stage 1)
CUDA_VISIBLE_DEVICES=0 ns-train nep --data ./data/bunny3 --pipeline.model.nep-cfg ' -pr -m nep1_mip360 shader_cfg.light_act=sigmoid nerf.rgb_exp=false shader_cfg.use_refneus=true rgb_loss=score_l1 shader_cfg.lr_decomp=true loss.curvature.weight=0.65 shader_cfg.human_light=false'  --max-num-iterations 300001 --pipeline.model.eval-num-rays-per-chunk 1800  --expname s1_360srnlr_c --pipeline.datamanager.train-num-rays-per-batch 512  nep-real-data --eval-images <img_path> --downscale-factor 1

# extract mesh
python3 nep/extract_mesh.py <stage1_ckpt_path>

# material learning (stage2)
CUDA_VISIBLE_DEVICES=0 ns-train nep --data ./data/bunny3 --pipeline.model.nep-cfg ' -pr -m nep2_mip_r1 shader_cfg.inner_light=trace_detach bg_model.freeze_nerf=true shader_cfg.outer_light_version=[sphere_direction,nerf] shader_cfg.outer_light_act=tanh shader_cfg.outer_light_reduce=sky shader_cfg.human_light=false bg_model.ckpt=log/s1_360Dcn@bunny3/300000.ckpt' --pipeline.mesh log/s1_360Dcn@bunny3/noclip-300000.ply --steps-per-eval-image=2500 --optimizers.network.scheduler.warmup-steps 1000 --optimizers.network.scheduler.max-steps 10001 --max-num-iterations 10001 --pipeline.model.eval-num-rays-per-chunk 1800  --expname s2_tri15+s1_360Dcn --pipeline.datamanager.train-num-rays-per-batch 512  nep-real-data --eval-images <img_path> --downscale-factor 1
```
