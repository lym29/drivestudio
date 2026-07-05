"""Render novel-view Background videos with optional custom trajectories."""

import argparse
import logging
import os

import torch
from omegaconf import OmegaConf

from datasets.driving_dataset import DrivingDataset
from models.video_utils import render_novel_background_views
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


def count_drive_frames(num_timesteps: int, drive_stride: int) -> int:
    mid_frame = num_timesteps // 2
    return len(range(0, mid_frame + 1, drive_stride))


def main():
    parser = argparse.ArgumentParser("Render novel Background views")
    parser.add_argument("--resume_from", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--traj_type", type=str, default="orbit_pullback")
    parser.add_argument("--pullback_m", type=float, default=0.0)
    parser.add_argument("--height_m", type=float, default=0.2)
    parser.add_argument("--height_axis_sign", type=float, default=-1.0)
    parser.add_argument("--elevate_frames", type=int, default=15)
    parser.add_argument("--spin_frames", type=int, default=360)
    parser.add_argument("--exit_transition_frames", type=int, default=20)
    parser.add_argument("--orbit_deg", type=float, default=360.0)
    parser.add_argument("--orbit_frames", type=int, default=120)
    parser.add_argument("--end_hold_frames", type=int, default=0)
    parser.add_argument("--drive_stride", type=int, default=1)
    parser.add_argument("--transition_frames", type=int, default=40)
    parser.add_argument("--orbit_radius", type=float, default=None)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer, dataset, _ = build_trainer_and_dataset(args.resume_from, device)

    traj_kwargs = {
        "scene_origin": trainer.scene_origin,
        "pullback_m": args.pullback_m,
        "height_m": args.height_m,
        "height_axis_sign": args.height_axis_sign,
        "elevate_frames": args.elevate_frames,
        "spin_frames": args.spin_frames,
        "exit_transition_frames": args.exit_transition_frames,
        "spin_in_place": True,
        "orbit_deg": args.orbit_deg,
        "orbit_frames": args.orbit_frames,
        "end_hold_frames": args.end_hold_frames,
        "drive_stride": args.drive_stride,
        "transition_frames": args.transition_frames,
    }
    if args.orbit_radius is not None:
        traj_kwargs["orbit_radius"] = args.orbit_radius

    traj = dataset.get_novel_render_traj(
        traj_types=[args.traj_type],
        target_frames=1,
        traj_kwargs=traj_kwargs,
    )[args.traj_type]
    render_data = dataset.prepare_novel_view_render_data(traj)
    drive_first_count = count_drive_frames(dataset.num_img_timesteps, args.drive_stride)
    spin_segment_count = (
        max(args.elevate_frames + args.spin_frames - 1, 0)
        + max(args.exit_transition_frames - 1, 0)
    )

    logger.info(
        "Rendering %s: total_frames=%d, drive_first=%d, spin_segment=%d, fps=%d",
        args.traj_type,
        len(render_data),
        drive_first_count,
        spin_segment_count,
        args.fps,
    )

    render_novel_background_views(
        trainer=trainer,
        render_data=render_data,
        save_path=args.output,
        fps=args.fps,
        drive_frame_count=drive_first_count,
        spin_segment_count=spin_segment_count,
    )


if __name__ == "__main__":
    main()
