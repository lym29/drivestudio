"""Render ego-driving videos with ego-speed simulation."""

import argparse
import logging
import os

import torch
from omegaconf import OmegaConf

from datasets.driving_dataset import DrivingDataset
from models.video_utils import render_novel_views
from utils.instance_utils import SPEED_PRESETS
from utils.misc import import_str

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_trainer_and_dataset(resume_from: str, device: torch.device):
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
    return trainer, dataset, cfg


def resolve_speed_factor(args) -> float:
    if args.speed_factor is not None:
        return args.speed_factor
    if args.preset is None:
        raise ValueError("Provide --speed_factor or --preset.")
    if args.preset not in SPEED_PRESETS:
        raise ValueError(f"Unknown preset {args.preset}. Options: {list(SPEED_PRESETS)}")
    return SPEED_PRESETS[args.preset]


def main():
    parser = argparse.ArgumentParser("Render ego-speed simulation videos")
    parser.add_argument("--resume_from", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--speed_factor",
        type=float,
        default=None,
        help="Ego speed multiplier (<1 slower, >1 faster)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=list(SPEED_PRESETS),
        help="normal=1.0, slow=0.7, fast=1.6",
    )
    parser.add_argument("--cam_id", type=int, default=0)
    parser.add_argument("--fps", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer, dataset, cfg = build_trainer_and_dataset(args.resume_from, device)

    if args.output is None and args.output_dir is None:
        raise ValueError("Provide --output or --output_dir.")

    ego_speed_factor = resolve_speed_factor(args)
    fps = args.fps if args.fps is not None else cfg.render.get("fps", 10)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        preset_name = args.preset or f"x{ego_speed_factor:g}"
        output = os.path.join(args.output_dir, f"ego_speed_{preset_name}.mp4")
    else:
        output = args.output
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    if "RigidNodes" in trainer.models:
        trainer.models["RigidNodes"].clear_instance_time_scales()

    render_data = dataset.prepare_ego_speed_render_data(
        cam_id=args.cam_id,
        ego_speed_factor=ego_speed_factor,
    )

    logger.info(
        "Rendering ego speed sim: ego_speed_factor=%.2f frames=%d fps=%d -> %s",
        ego_speed_factor,
        len(render_data),
        fps,
        output,
    )

    render_novel_views(
        trainer=trainer,
        render_data=render_data,
        save_path=output,
        fps=fps,
    )
    logger.info("Saved: %s", output)


if __name__ == "__main__":
    main()
