"""Export Background Gaussian Splatting model from checkpoint to PLY."""

import argparse
import logging
import os
import sys
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from datasets.driving_dataset import DrivingDataset
from utils.misc import export_gaussians_to_ply, import_str

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCENE_IDS = ["000", "001", "002", "003", "004", "005", "006", "008", "009"]
DEFAULT_OUTPUT_ROOT = "/DATA/lym_data/omnire-results"
DEFAULT_CHECKPOINT_ROOT = "/DATA/OmniRe-output/recon"
PROJECT_NAME = "exp-nusences-mini"


class BackgroundGaussianAdapter:
    """Adapt VanillaGaussians to the interface expected by export_gaussians_to_ply."""

    def __init__(self, model):
        self._model = model
        self.config = SimpleNamespace(sh_degree=model.sh_degree)

    def eval(self):
        self._model.eval()
        return self

    @property
    def means(self):
        return self._model._means

    @property
    def colors(self):
        return self._model.colors

    @property
    def shs_rest(self):
        return self._model._features_rest

    @property
    def opacities(self):
        return self._model._opacities

    @property
    def scales(self):
        return self._model._scales

    @property
    def quats(self):
        return self._model._quats


def load_trainer(resume_from: str, device: torch.device):
    log_dir = os.path.dirname(resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))

    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    trainer.resume_from_checkpoint(ckpt_path=resume_from, load_only_model=True)
    return trainer, cfg


def export_background_ply(resume_from: str, output_dir: str, device: torch.device, world_coords: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    trainer, _ = load_trainer(resume_from, device)

    if "Background" not in trainer.models:
        raise KeyError(f"Background model not found in checkpoint: {resume_from}")

    background_model = BackgroundGaussianAdapter(trainer.models["Background"])
    ply_name = "background_world.ply" if world_coords else "background.ply"
    export_gaussians_to_ply(background_model, output_dir, name=ply_name, world_coords=world_coords)
    logger.info("Exported Background PLY from %s to %s", resume_from, output_dir)


def resolve_checkpoint_path(scene_id: str, checkpoint_root: str) -> str:
    return os.path.join(
        checkpoint_root, f"{PROJECT_NAME}-{scene_id}", "checkpoint_final.pth"
    )


def main():
    parser = argparse.ArgumentParser("Export Background 3DGS to PLY")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint_final.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save background.ply",
    )
    parser.add_argument(
        "--scene_ids",
        nargs="+",
        default=None,
        help=f"Scene IDs for batch export (default: all completed scenes)",
    )
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Root directory containing exp-nusences-mini-XXX folders",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for organized omnire-results",
    )
    parser.add_argument(
        "--world_coords",
        action="store_true",
        help="Export positions in world coordinates instead of normalized unit ball",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.resume_from:
        if not args.output_dir:
            parser.error("--output_dir is required when using --resume_from")
        export_background_ply(args.resume_from, args.output_dir, device, world_coords=args.world_coords)
        return

    scene_ids = args.scene_ids or SCENE_IDS
    failed = []
    for scene_id in scene_ids:
        resume_from = resolve_checkpoint_path(scene_id, args.checkpoint_root)
        output_dir = os.path.join(args.output_root, f"scene_{scene_id}", "models")

        if not os.path.isfile(resume_from):
            logger.warning("Skip scene %s: checkpoint not found at %s", scene_id, resume_from)
            failed.append(scene_id)
            continue

        try:
            export_background_ply(resume_from, output_dir, device, world_coords=args.world_coords)
        except Exception as exc:
            logger.error("Failed to export scene %s: %s", scene_id, exc)
            failed.append(scene_id)

    if failed:
        logger.error("Failed scenes: %s", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
