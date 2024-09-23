import shutil
import trimesh
from nep.network.fields.utils import extract_geometry
from nep.utils.raw_utils import linear_to_srgb
from nerfstudio.engine.trainer import Trainer, TrainerConfig, TRAIN_INTERATION_OUTPUT
from typing import Optional, Type, Literal
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName
from nep.ns.buffer import GLOBALS
import torch
import os
from dataclasses import dataclass, field
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils import profiler, writer
import time
import yaml
from nerfstudio.utils.decorators import (
    check_eval_enabled,
    check_main_thread,
    check_viewer_enabled,
)
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm
from nep.ns.models.nep import parse_arch_from_nep_str
from nep.network.renderers.material import NePMaterialRenderer
from setproctitle import setproctitle
import nerfstudio


import numpy as np
import os

from nep.network.renderers.shape import NePShapeRenderer


@dataclass
class MyTrainerConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: MyTrainer)
    # tag: Optional[str] = None
    debug: bool = False
    proc: bool = False
    skip_save_ckpt: bool = False
    expname: Optional[str] = None
    autobreak: bool = False
    npz: bool = False
    extract_material: bool = False
    gt_roughness: float = 0.1
    gt_metallic: float = 1.0
    gt_albedo: list = field(default_factory=lambda: [0.5, 0.5, 0.5])
    output_dir: Path = Path("log")
    relative_model_dir: str = "models"

    def get_checkpoint_dir(self) -> Path:
        """Retrieve the checkpoint directory"""
        return self.get_base_dir()

    def get_base_dir(self) -> Path:
        """Retrieve the base directory to set relative paths"""
        # check the experiment and method names
        assert self.method_name is not None, "Please set method name in config or via the cli"
        self.set_experiment_name()
        return Path(f"{self.output_dir}/{self.expname}@{self.experiment_name}")


class MyTrainer(Trainer):
    config: MyTrainerConfig

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        print(f"Abs: {self.base_dir.absolute()}")
        if hasattr(self.config.pipeline, "mesh") and self.config.pipeline.mesh is not None:
            GLOBALS["meshdir"] = Path(self.config.pipeline.mesh).parent

        # load latest ckpt
        # if not self.config.debug:
        load_dir = self.base_dir
        if len(list(load_dir.glob("*.ckpt"))) > 0:
            self.config.load_dir = load_dir

        super().setup(test_mode)

        # set init step for model and pbar
        if self.config.npz:
            self.npz_data = None
            print("npz path:", self.base_dir / "training_log.npz")

        self.config.max_num_iterations = self.config.max_num_iterations - self._start_step
        self.pbar = tqdm(
            initial=self._start_step,
            total=self._start_step + self.config.max_num_iterations,
            bar_format="{r_bar}",
        )

        # for nep
        if hasattr(self.pipeline.model, "train_step"):
            self.pipeline.model.train_step = self._start_step
            self.pipeline.model.total_step = self.config.max_num_iterations

        # save config and log
        if hasattr(self.pipeline.model, "nep_config"):
            with open(self.base_dir / "nep-config.yaml", "w") as file:
                yaml.dump(self.pipeline.model.nep_config, file)

        # set proc name
        if self.config.proc:
            print("Using custom proc name.")
            setproctitle("python launch_test.py --train dataset.scene=../../data/test_data")

    @check_viewer_enabled
    def _train_complete_viewer(self) -> None:
        """
        copied from Trainer._train_complete_viewer, but:
            - remove the inf loop to wait user confirm.
        """
        assert self.viewer_state is not None
        print()
        self.training_state = "completed"
        try:
            self.viewer_state.training_complete()
        except RuntimeError:
            time.sleep(0.03)  # sleep to allow buffer to reset
            CONSOLE.log("Viewer failed. Continuing training.")

    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        if self.config.autobreak and step - self._start_step == 10001:
            print("Autobreak")
            os._exit(0)

        loss, loss_dict, metrics_dict = super().train_iteration(step)
        self.pbar.set_postfix(
            {k.replace("loss_", "").replace("_loss", ""): float(v) for k, v in loss_dict.items()}
        )
        self.pbar.update(1)

        # close the bar to make the following messages printed to the next line.
        if step == self.config.max_num_iterations - 1:
            self.pbar.close()

        # log data
        if not self.config.debug and self.config.npz:
            self.log_to_npz(loss_dict, metrics_dict, filename=self.base_dir / "training_log.npz")
        return loss, loss_dict, metrics_dict

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        # extract material only and exit
        if step_check(step, self.config.steps_per_eval_image) and (
            hasattr(self.pipeline.model, "network")
            and isinstance(self.pipeline.model.network, NePMaterialRenderer)
            and (self.config.extract_material or step >= self.config.max_num_iterations - 10)
        ):
            materials = self.pipeline.predict_materials()
            material_dir = self.base_dir / "materials"
            material_dir.mkdir(exist_ok=True)

            mapping_fn = lambda x: x
            # mapping_fn = linear_to_srgb
            np.save(f"{material_dir}/metallic.npy", mapping_fn(materials["metallic"]))
            np.save(f"{material_dir}/roughness.npy", mapping_fn(materials["roughness"]))
            np.save(f"{material_dir}/albedo.npy", mapping_fn(materials["albedo"]))

            # print the MSE
            assert (
                self.config.gt_roughness is not None
                and self.config.gt_metallic is not None
                and self.config.gt_albedo is not None
            )

            def log_fn(file):
                print(
                    "GT roughness / metallic / albedo:",
                    self.config.gt_roughness,
                    self.config.gt_metallic,
                    self.config.gt_albedo,
                    file=file,
                )
                # print max min mean of each value
                print(
                    "roughness max min mean:",
                    materials["roughness"].max(),
                    materials["roughness"].min(),
                    materials["roughness"].mean(),
                    file=file,
                )
                print(
                    "metallic max min mean:",
                    materials["metallic"].max(),
                    materials["metallic"].min(),
                    materials["metallic"].mean(),
                    file=file,
                )
                print(
                    "albedo max min mean:",
                    materials["albedo"].max(axis=0),
                    materials["albedo"].min(axis=0),
                    materials["albedo"].mean(axis=0),
                    file=file,
                )

                print("MSE roughness / metallic / albedo:", file=file)
                print(
                    np.mean((materials["roughness"] - self.config.gt_roughness) ** 2),
                    np.mean((materials["metallic"] - self.config.gt_metallic) ** 2),
                    np.mean((materials["albedo"] - np.array(self.config.gt_albedo) / 255) ** 2),
                    file=file,
                )

            log_fn(None)
            log_fn(open("outputs/mat.txt", "a"))

            for item in writer.EVENT_STORAGE:
                if not item["write_type"] == writer.EventType.IMAGE:
                    continue
                vis_name = item["name"].split("/")[-1]
                if "vis" in vis_name:
                    continue
                backup_dir = self.config.output_dir / "materials"
                backup_dir.mkdir(exist_ok=True)
                savepath = (
                    backup_dir
                    / f"{self.config.expname}@{self.config.experiment_name}-{vis_name}.jpg"
                )
                print("Save image:", savepath.absolute())
                save_image(item["event"].permute(-1, 0, 1), savepath)

            if self.config.extract_material:
                os._exit(0)

        if self.config.debug:
            super().eval_iteration(step)
        else:
            try:
                super().eval_iteration(step)
            except Exception as e:
                print(f"Got error in eval_iteration, skip:")
                print(e)

        if step_check(step, self.config.steps_per_eval_image):
            # create vis dir
            # vis_dir = self.base_dir / "vis"
            vis_dir = self.base_dir
            if not vis_dir.exists():
                vis_dir.mkdir()

                # save images
            for item in writer.EVENT_STORAGE:
                if not item["write_type"] == writer.EventType.IMAGE:
                    continue
                vis_name = item["name"].split("/")[-1]
                if "vis" not in "vis_name":
                    continue
                savepath = vis_dir / f"{vis_name}-{step}.jpg"
                print("Save image:", savepath.absolute())
                save_image(item["event"].permute(-1, 0, 1), savepath)

                # add a backup
                backup_dir = self.config.output_dir / "vis"
                backup_dir.mkdir(exist_ok=True)
                shutil.copy(
                    savepath,
                    backup_dir
                    / f"{self.config.expname}@{self.config.experiment_name}-{vis_name}-latest.jpg",
                )

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        if step < 100 or self.config.skip_save_ckpt:
            return
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path: Path = self.checkpoint_dir / f"{step}.ckpt"
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*.ckpt"):
                if f != ckpt_path:
                    f.unlink()

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest Nerfstudio checkpoint from load_dir...")
                # NOTE: this is specific to the checkpoint name format
                filelist = [x.name for x in load_dir.glob("*.ckpt")]
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in filelist)[-1]

            # load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            load_path: Path = load_dir / f"{load_step}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")

        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")
