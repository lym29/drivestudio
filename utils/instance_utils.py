"""Utilities for inspecting RigidNodes instances and finding the lead vehicle."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch

from datasets.base.scene_dataset import ModelType

logger = logging.getLogger(__name__)

SPEED_PRESETS = {
    "normal": 1.0,   # ego at original driving speed
    "slow": 0.7,     # ego moderately slower -> traffic appears a bit faster
    "fast": 1.6,     # ego moderately faster -> traffic appears a bit slower
}


def _load_class_name_map(pixel_source) -> Dict[int, str]:
    path = os.path.join(pixel_source.data_path, "instances", "instances_info.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        instances_info = json.load(f)
    return {
        int(k): v.get("class_name", "unknown")
        for k, v in instances_info.items()
    }


def get_rigid_model_to_dataset_ids(trainer, dataset) -> Dict[int, int]:
    """Map RigidNodes model index to dataset instance id."""
    rigid = trainer.models.get("RigidNodes")
    if rigid is None:
        return {}

    if (
        hasattr(rigid, "dataset_instance_ids")
        and rigid.dataset_instance_ids is not None
        and len(rigid.dataset_instance_ids) > 0
    ):
        return {
            i: int(rigid.dataset_instance_ids[i].item())
            for i in range(len(rigid.dataset_instance_ids))
        }

    if "RigidNodes" not in trainer.model_config:
        return {}

    instance_dict = dataset.get_init_objects(
        cur_node_type="RigidNodes",
        **trainer.model_config["RigidNodes"]["init"],
    )
    return {model_id: int(dataset_id) for model_id, dataset_id in enumerate(instance_dict.keys())}


def _trajectory_length(pixel_source, dataset_id: int) -> float:
    frame_info = pixel_source.per_frame_instance_mask[:, dataset_id]
    poses = pixel_source.instances_pose[:, dataset_id]
    valid_trans = poses[:, :3, 3][frame_info]
    if len(valid_trans) < 2:
        return 0.0
    deltas = valid_trans[1:] - valid_trans[:-1]
    return float(torch.norm(deltas, dim=-1).sum().item())


def list_rigid_instances(trainer, dataset) -> List[Dict]:
    """Return metadata for each RigidNodes instance."""
    rigid = trainer.models.get("RigidNodes")
    if rigid is None:
        return []

    model_to_dataset = get_rigid_model_to_dataset_ids(trainer, dataset)
    class_names = _load_class_name_map(dataset.pixel_source)
    instances = []

    for model_id, dataset_id in model_to_dataset.items():
        true_id = int(dataset.pixel_source.instances_true_id[dataset_id].item())
        class_name = class_names.get(true_id, "unknown")
        traj_len = _trajectory_length(dataset.pixel_source, dataset_id)
        num_pts = int((rigid.point_ids[..., 0] == model_id).sum().item())
        instances.append(
            {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "true_id": true_id,
                "class_name": class_name,
                "traj_length": traj_len,
                "num_pts": num_pts,
            }
        )
    return instances


def print_rigid_instances(trainer, dataset) -> None:
    instances = list_rigid_instances(trainer, dataset)
    if not instances:
        logger.info("No RigidNodes instances found.")
        return
    logger.info("RigidNodes instances:")
    for ins in instances:
        logger.info(
            "  model_id=%d dataset_id=%d true_id=%d class=%s traj_len=%.2f num_pts=%d",
            ins["model_id"],
            ins["dataset_id"],
            ins["true_id"],
            ins["class_name"],
            ins["traj_length"],
            ins["num_pts"],
        )


def _world_to_cam(pos_world: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    w2c = torch.linalg.inv(c2w)
    pos_h = torch.cat([pos_world, torch.ones(1, device=pos_world.device)])
    return (w2c @ pos_h)[:3]


def find_lead_vehicle(
    trainer,
    dataset,
    cam_id: int = 0,
    frame_idx: Optional[int] = None,
    min_depth: float = 3.0,
    max_lateral: float = 15.0,
) -> Tuple[int, Dict]:
    """
    Find the lead vehicle RigidNodes model index.

    Picks the visible RigidNodes vehicle directly ahead of the front camera
    with the smallest positive depth at the reference frame.
    """
    rigid = trainer.models.get("RigidNodes")
    if rigid is None:
        raise RuntimeError("RigidNodes model not found in checkpoint.")

    model_to_dataset = get_rigid_model_to_dataset_ids(trainer, dataset)
    pixel_source = dataset.pixel_source
    if frame_idx is None:
        frame_idx = pixel_source.num_frames // 2

    cam = pixel_source.camera_data[cam_id]
    c2w = cam.cam_to_worlds[frame_idx].to(pixel_source.device)

    best_model_id = None
    best_depth = float("inf")
    best_info = {}

    for model_id, dataset_id in model_to_dataset.items():
        if not pixel_source.per_frame_instance_mask[frame_idx, dataset_id]:
            continue
        if pixel_source.instances_model_types[dataset_id].item() != ModelType.RigidNodes:
            continue

        pos_world = pixel_source.instances_pose[frame_idx, dataset_id, :3, 3]
        pos_cam = _world_to_cam(pos_world, c2w)
        depth = float(pos_cam[2].item())
        lateral = float(abs(pos_cam[0].item()))

        if depth <= min_depth or lateral > max_lateral:
            continue
        if depth < best_depth:
            best_depth = depth
            best_model_id = model_id
            best_info = {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "frame_idx": frame_idx,
                "depth": depth,
                "lateral": lateral,
                "world_pos": pos_world.detach().cpu().tolist(),
            }

    if best_model_id is None:
        raise RuntimeError(
            f"Could not auto-detect lead vehicle at frame {frame_idx}. "
            "Use --instance_id to specify manually."
        )

    logger.info(
        "Auto-detected lead vehicle: model_id=%d dataset_id=%d depth=%.2fm lateral=%.2fm",
        best_info["model_id"],
        best_info["dataset_id"],
        best_info["depth"],
        best_info["lateral"],
    )
    return best_model_id, best_info
