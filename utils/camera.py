# Camera pose manipulation and trajectory generation.
import os
from typing import Dict, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

WORLD_UP = torch.tensor([0.0, 1.0, 0.0])


def detect_world_up(per_cam_poses: Dict[int, torch.Tensor]) -> torch.Tensor:
    """Infer world up from the average camera-up column in c2w matrices."""
    poses = per_cam_poses[0]
    up = poses[:, :3, 1].mean(dim=0)
    return torch.nn.functional.normalize(up, dim=0)

def interpolate_poses(key_poses: torch.Tensor, target_frames: int) -> torch.Tensor:
    """
    Interpolate between key poses to generate a smooth trajectory.
    
    Args:
        key_poses (torch.Tensor): Tensor of shape (N, 4, 4) containing key camera poses.
        target_frames (int): Number of frames to interpolate.
    
    Returns:
        torch.Tensor: Interpolated poses of shape (target_frames, 4, 4).
    """
    device = key_poses.device
    key_poses = key_poses.cpu().numpy()
    
    # Separate translation and rotation
    translations = key_poses[:, :3, 3]
    rotations = key_poses[:, :3, :3]
    
    # Create time array
    times = np.linspace(0, 1, len(key_poses))
    target_times = np.linspace(0, 1, target_frames)
    
    # Interpolate translations
    interp_translations = np.stack([
        np.interp(target_times, times, translations[:, i])
        for i in range(3)
    ], axis=-1)
    
    # Interpolate rotations using Slerp
    key_rots = R.from_matrix(rotations)
    slerp = Slerp(times, key_rots)
    interp_rotations = slerp(target_times).as_matrix()
    
    # Combine interpolated translations and rotations
    interp_poses = np.eye(4)[None].repeat(target_frames, axis=0)
    interp_poses[:, :3, :3] = interp_rotations
    interp_poses[:, :3, 3] = interp_translations
    
    return torch.tensor(interp_poses, dtype=torch.float32, device=device)

def look_at_rotation(direction: torch.Tensor, up: torch.Tensor = torch.tensor([0., 0., 1.])) -> torch.Tensor:
    """Calculate rotation matrix to look at a specific direction."""
    front = torch.nn.functional.normalize(direction, dim=-1)
    right = torch.nn.functional.normalize(torch.cross(front, up), dim=-1)
    up = torch.cross(right, front)
    rotation_matrix = torch.stack([right, up, -front], dim=-1)
    return rotation_matrix


def _smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _blend_rotations(
    start_rot: torch.Tensor,
    end_rot: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """Spherical interpolation between two rotation matrices."""
    t = _smoothstep(t)
    key_rots = R.from_matrix(
        torch.stack([start_rot, end_rot], dim=0).cpu().numpy()
    )
    slerp = Slerp([0.0, 1.0], key_rots)
    return torch.tensor(
        slerp(t).as_matrix(),
        dtype=start_rot.dtype,
        device=start_rot.device,
    )

def get_interp_novel_trajectories(
    dataset_type: str,
    scene_idx: str,
    per_cam_poses: Dict[int, torch.Tensor],
    traj_type: str = "front_center_interp",
    target_frames: int = 100,
    traj_kwargs: Optional[Dict] = None,
) -> torch.Tensor:
    original_frames = per_cam_poses[list(per_cam_poses.keys())[0]].shape[0]
    traj_kwargs = traj_kwargs or {}

    trajectory_generators = {
        "front_center_interp": front_center_interp,
        "s_curve": s_curve,
        "three_key_poses": three_key_poses_trajectory,
        "orbit_pullback": orbit_pullback_trajectory,
    }

    if traj_type not in trajectory_generators:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    if traj_type == "orbit_pullback":
        return trajectory_generators[traj_type](
            dataset_type, per_cam_poses, original_frames, target_frames, traj_kwargs
        )

    return trajectory_generators[traj_type](
        dataset_type, per_cam_poses, original_frames, target_frames
    )

def front_center_interp(
    dataset_type: str, per_cam_poses: Dict[int, torch.Tensor], original_frames: int, target_frames: int, num_loops: int = 1
) -> torch.Tensor:
    """Interpolate key frames from the front center camera."""
    assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required for front_center_interp"
    key_poses = per_cam_poses[0][::original_frames//4]  # Select every 4th frame as key frame
    return interpolate_poses(key_poses, target_frames)

def s_curve(
    dataset_type: str, per_cam_poses: Dict[int, torch.Tensor], original_frames: int, target_frames: int
) -> torch.Tensor:
    """Create an S-shaped trajectory using the front three cameras."""
    assert all(cam in per_cam_poses.keys() for cam in [0, 1, 2]), "Front three cameras (IDs 0, 1, 2) are required for s_curve"
    key_poses = torch.cat([
        per_cam_poses[0][0:1],
        per_cam_poses[1][original_frames//4:original_frames//4+1],
        per_cam_poses[0][original_frames//2:original_frames//2+1],
        per_cam_poses[2][3*original_frames//4:3*original_frames//4+1],
        per_cam_poses[0][-1:]
    ], dim=0)
    return interpolate_poses(key_poses, target_frames)

def three_key_poses_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int
) -> torch.Tensor:
    """
    Create a trajectory using three key poses:
    1. First frame of front center camera
    2. Middle frame with interpolated rotation and position from camera 1 or 2
    3. Last frame of front center camera

    The rotation of the middle pose is calculated using Slerp between
    the start frame and the middle frame of camera 1 or 2.

    Args:
        dataset_type (str): Type of the dataset (e.g., "waymo", "pandaset", etc.).
        per_cam_poses (Dict[int, torch.Tensor]): Dictionary of camera poses.
        original_frames (int): Number of original frames.
        target_frames (int): Number of frames in the output trajectory.

    Returns:
        torch.Tensor: Trajectory of shape (target_frames, 4, 4).
    """
    assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required"
    assert 1 in per_cam_poses.keys() or 2 in per_cam_poses.keys(), "Either camera 1 or camera 2 is required"

    # First key pose: First frame of front center camera
    start_pose = per_cam_poses[0][0]
    key_poses = [start_pose]

    # Select camera for middle frame
    middle_frame = int(original_frames // 2)
    chosen_cam = np.random.choice([1, 2])

    middle_pose = per_cam_poses[chosen_cam][middle_frame]

    # Calculate interpolated rotation for middle pose
    start_rotation = R.from_matrix(start_pose[:3, :3].cpu().numpy())
    middle_rotation = R.from_matrix(middle_pose[:3, :3].cpu().numpy())
    slerp = Slerp([0, 1], R.from_quat([start_rotation.as_quat(), middle_rotation.as_quat()]))
    interpolated_rotation = slerp(0.5).as_matrix()

    # Create middle key pose with interpolated rotation and original translation
    middle_key_pose = torch.eye(4, device=start_pose.device)
    middle_key_pose[:3, :3] = torch.tensor(interpolated_rotation, device=start_pose.device)
    middle_key_pose[:3, 3] = middle_pose[:3, 3]  # Keep the original translation
    key_poses.append(middle_key_pose)

    # Third key pose: Last frame of front center camera
    key_poses.append(per_cam_poses[0][-1])

    # Stack the key poses and interpolate
    key_poses = torch.stack(key_poses)
    return interpolate_poses(key_poses, target_frames)


def pose_from_position_lookat(
    position: torch.Tensor,
    look_at: torch.Tensor,
    up: torch.Tensor = WORLD_UP,
) -> torch.Tensor:
    direction = look_at - position
    rot = look_at_rotation(direction, up)
    pose = torch.eye(4, device=position.device, dtype=position.dtype)
    pose[:3, :3] = rot
    pose[:3, 3] = position
    return pose


def _horizontal_components(
    vec: torch.Tensor, world_up: torch.Tensor
) -> torch.Tensor:
    """Return the horizontal part of vec (perpendicular to world_up)."""
    return vec - world_up * torch.dot(vec, world_up)


def _ground_position(
    position: torch.Tensor, ground_height: float, world_up: torch.Tensor
) -> torch.Tensor:
    """Set position height along world_up to ground_height."""
    height = torch.dot(position, world_up)
    return position + world_up * (ground_height - height)


def generate_junction_yaw_spin(
    junction_c2w: torch.Tensor,
    height_m: float,
    elevate_frames: int,
    spin_frames: int,
    world_up: torch.Tensor = WORLD_UP,
    height_axis_sign: float = -1.0,
) -> torch.Tensor:
    """
    Elevate at the junction pose, then spin 360° around world +Y in place.

    ``height_axis_sign`` controls the elevation direction along ``world_up``.
    Use ``-1`` when the scene's visual up requires a negative Y offset.
    """
    device = junction_c2w.device
    world_up = world_up.to(device).float()
    junction_pos = junction_c2w[:3, 3].float()
    junction_rot = junction_c2w[:3, :3]
    height_offset = world_up * (height_m * height_axis_sign)
    elevated_pos = junction_pos + height_offset

    poses = [junction_c2w.clone()]

    for i in range(1, max(elevate_frames, 1)):
        t = _smoothstep(i / max(elevate_frames - 1, 1))
        pose = torch.eye(4, device=device, dtype=junction_c2w.dtype)
        pose[:3, :3] = junction_rot
        pose[:3, 3] = junction_pos + height_offset * t
        poses.append(pose)

    spin_rad = 2.0 * np.pi
    for i in range(max(spin_frames, 1)):
        spin_t = _smoothstep(i / max(spin_frames - 1, 1))
        angle = spin_rad * spin_t
        yaw_rot = R.from_rotvec(world_up.cpu().numpy() * angle).as_matrix()
        rot = torch.tensor(yaw_rot, dtype=junction_rot.dtype, device=device) @ junction_rot
        pose = torch.eye(4, device=device, dtype=junction_c2w.dtype)
        pose[:3, :3] = rot
        pose[:3, 3] = elevated_pos
        poses.append(pose)

    return torch.stack(poses, dim=0)


def _blend_poses(
    start_pose: torch.Tensor,
    end_pose: torch.Tensor,
    t: float,
) -> torch.Tensor:
    pose = torch.eye(4, device=start_pose.device, dtype=start_pose.dtype)
    pose[:3, :3] = _blend_rotations(start_pose[:3, :3], end_pose[:3, :3], t)
    pose[:3, 3] = start_pose[:3, 3] * (1.0 - t) + end_pose[:3, 3] * t
    return pose


def generate_horizontal_orbit(
    start_c2w: torch.Tensor,
    orbit_center: torch.Tensor,
    height_m: float,
    orbit_deg: float,
    n_frames: int,
    world_up: torch.Tensor = WORLD_UP,
    orbit_radius: Optional[float] = None,
    transition_frames: int = 40,
) -> torch.Tensor:
    """
    Orbit horizontally around a scene center with a slight +Y elevation.

    The first frame matches the driving junction pose exactly, then position and
    rotation ease into the orbit over ``transition_frames``.
    """
    device = start_c2w.device
    orbit_center = orbit_center.to(device).float()
    world_up = world_up.to(device).float()
    start_pos = start_c2w[:3, 3].float()
    start_rot = start_c2w[:3, :3]

    offset = _horizontal_components(start_pos - orbit_center, world_up)
    radius = torch.norm(offset)
    if orbit_radius is not None and orbit_radius > 0:
        radius = torch.tensor(float(orbit_radius), device=device)
    elif radius < 1e-3:
        radius = torch.tensor(15.0, device=device)
        offset = torch.tensor([radius, 0.0, 0.0], device=device)

    start_angle = torch.atan2(offset[2], offset[0])
    look_at = orbit_center.clone()
    look_at[1] = start_pos[1]

    orbit_rad = np.deg2rad(orbit_deg)
    blend_frames = max(int(transition_frames), 1)

    poses = []
    for i in range(n_frames):
        progress = i / max(n_frames - 1, 1)
        enter_t = _smoothstep(min(1.0, i / max(blend_frames - 1, 1)))
        orbit_t = progress * enter_t

        angle = start_angle + orbit_rad * orbit_t
        radial = torch.stack(
            [torch.cos(angle), torch.tensor(0.0, device=device), torch.sin(angle)]
        )
        orbit_pos = orbit_center + radial * radius
        orbit_pos[1] = start_pos[1] + height_m * enter_t
        position = start_pos * (1.0 - enter_t) + orbit_pos * enter_t

        target_rot = pose_from_position_lookat(position, look_at, world_up)[:3, :3]
        rot = _blend_rotations(start_rot, target_rot, enter_t)

        pose = torch.eye(4, device=device, dtype=start_rot.dtype)
        pose[:3, :3] = rot
        pose[:3, 3] = position
        poses.append(pose)

    poses[0] = start_c2w.clone()
    return torch.stack(poses, dim=0)


def generate_pullback_rise_arc(
    start_c2w: torch.Tensor,
    pullback_m: float,
    height_m: float,
    orbit_deg: float,
    n_frames: int,
    world_up: torch.Tensor = WORLD_UP,
    orbit_lateral_scale: float = 0.35,
    yaw_scale: float = 0.4,
) -> torch.Tensor:
    """
    Pull back, rise, and arc laterally from the driving junction pose.

    The camera keeps the driving orientation (no look-at-to-ground), so the
    view stays near the horizon instead of tilting downward.
    """
    device = start_c2w.device
    world_up = world_up.to(device).float()
    start_pos = start_c2w[:3, 3].float()
    start_rot = start_c2w[:3, :3]

    start_forward = _horizontal_components(-start_c2w[:3, 2], world_up)
    if torch.norm(start_forward) < 1e-3:
        start_forward = torch.tensor([0.0, 0.0, 1.0], device=device)
    start_forward = torch.nn.functional.normalize(start_forward, dim=0)
    start_right = torch.nn.functional.normalize(
        torch.cross(world_up, start_forward), dim=0
    )

    orbit_rad = np.deg2rad(orbit_deg)
    angles = torch.linspace(0.0, orbit_rad, n_frames, device=device)

    poses = []
    for angle_delta in angles:
        progress = (
            float(angle_delta / orbit_rad) if orbit_rad > 0 else 0.0
        )
        motion_t = _smoothstep(progress)

        position = (
            start_pos
            - start_forward * (pullback_m * motion_t)
            + start_right * (pullback_m * orbit_lateral_scale * torch.sin(angle_delta))
            + world_up * (height_m * motion_t)
        )

        yaw_angle = float(angle_delta) * yaw_scale
        yaw_rot = R.from_rotvec(world_up.cpu().numpy() * yaw_angle).as_matrix()
        rot = torch.tensor(
            yaw_rot, dtype=start_rot.dtype, device=device
        ) @ start_rot

        pose = torch.eye(4, device=device, dtype=start_rot.dtype)
        pose[:3, :3] = rot
        pose[:3, 3] = position
        poses.append(pose)

    return torch.stack(poses, dim=0)


def generate_orbit_arc(
    start_c2w: torch.Tensor,
    orbit_center: torch.Tensor,
    look_at_start: torch.Tensor,
    look_at_end: torch.Tensor,
    pullback_m: float,
    height_m: float,
    orbit_deg: float,
    n_frames: int,
    world_up: torch.Tensor = WORLD_UP,
) -> torch.Tensor:
    """Deprecated path kept for compatibility; prefer generate_pullback_rise_arc."""
    return generate_pullback_rise_arc(
        start_c2w=start_c2w,
        pullback_m=pullback_m,
        height_m=height_m,
        orbit_deg=orbit_deg,
        n_frames=n_frames,
        world_up=world_up,
    )


def orbit_pullback_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    traj_kwargs: Dict,
) -> torch.Tensor:
    """
    Drive the first half, elevate and spin in place at the junction, then
    resume the original driving trajectory for the second half.
    """
    assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required for orbit_pullback"

    device = per_cam_poses[0].device
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)

    height_m = float(traj_kwargs.get("height_m", 0.1))
    height_axis_sign = float(traj_kwargs.get("height_axis_sign", -1.0))
    drive_stride = int(traj_kwargs.get("drive_stride", 1))
    elevate_frames = int(traj_kwargs.get("elevate_frames", 15))
    spin_frames = int(traj_kwargs.get("spin_frames", 90))
    exit_transition_frames = int(traj_kwargs.get("exit_transition_frames", 20))
    spin_in_place = bool(traj_kwargs.get("spin_in_place", True))
    pullback_m = float(traj_kwargs.get("pullback_m", 0.0))

    mid_frame = original_frames // 2
    drive_first = per_cam_poses[0][: mid_frame + 1 : drive_stride]
    drive_second = per_cam_poses[0][mid_frame + 1 :: drive_stride]
    junction_c2w = drive_first[-1]

    if spin_in_place and pullback_m <= 0:
        spin_poses = generate_junction_yaw_spin(
            junction_c2w=junction_c2w,
            height_m=height_m,
            elevate_frames=elevate_frames,
            spin_frames=spin_frames,
            world_up=world_up,
            height_axis_sign=height_axis_sign,
        )

        exit_blends = []
        spin_end = spin_poses[-1]
        resume_pose = drive_second[0]
        for i in range(max(exit_transition_frames, 1)):
            t = _smoothstep(i / max(exit_transition_frames - 1, 1))
            exit_blends.append(_blend_poses(spin_end, resume_pose, t))
        exit_blends = torch.stack(exit_blends, dim=0)

        return torch.cat(
            [drive_first, spin_poses[1:], exit_blends[1:], drive_second[1:]],
            dim=0,
        )

    scene_origin = traj_kwargs.get("scene_origin", None)
    if scene_origin is None:
        orbit_center = junction_c2w[:3, 3]
    elif isinstance(scene_origin, torch.Tensor):
        orbit_center = scene_origin.float().to(device)
    else:
        orbit_center = torch.tensor(scene_origin, dtype=torch.float32, device=device)

    orbit_deg = float(traj_kwargs.get("orbit_deg", 360.0))
    orbit_frames = int(traj_kwargs.get("orbit_frames", 120))
    end_hold_frames = int(traj_kwargs.get("end_hold_frames", 0))
    transition_frames = int(traj_kwargs.get("transition_frames", 40))
    orbit_radius = traj_kwargs.get("orbit_radius", None)
    if orbit_radius is not None:
        orbit_radius = float(orbit_radius)

    if pullback_m > 0:
        orbit_poses = generate_pullback_rise_arc(
            start_c2w=junction_c2w,
            pullback_m=pullback_m,
            height_m=height_m,
            orbit_deg=orbit_deg,
            n_frames=orbit_frames,
            world_up=world_up,
            orbit_lateral_scale=float(traj_kwargs.get("orbit_lateral_scale", 0.35)),
            yaw_scale=float(traj_kwargs.get("yaw_scale", 0.4)),
        )
    else:
        orbit_poses = generate_horizontal_orbit(
            start_c2w=junction_c2w,
            orbit_center=orbit_center,
            height_m=height_m,
            orbit_deg=orbit_deg,
            n_frames=orbit_frames,
            world_up=world_up,
            orbit_radius=orbit_radius,
            transition_frames=transition_frames,
        )

    if end_hold_frames > 0:
        end_hold = orbit_poses[-1:].repeat(end_hold_frames, 1, 1)
        return torch.cat([drive_first, orbit_poses[1:], end_hold], dim=0)

    return torch.cat([drive_first, orbit_poses[1:]], dim=0)


def interp_c2w_pair(c2w_lo: torch.Tensor, c2w_hi: torch.Tensor, alpha: float) -> torch.Tensor:
    """Interpolate between two camera-to-world poses."""
    if alpha < 1e-6:
        return c2w_lo
    if alpha > 1.0 - 1e-6:
        return c2w_hi

    device = c2w_lo.device
    lo = c2w_lo.detach().cpu().numpy()
    hi = c2w_hi.detach().cpu().numpy()
    trans = lo[:3, 3] * (1.0 - alpha) + hi[:3, 3] * alpha
    rots = np.stack([lo[:3, :3], hi[:3, :3]])
    slerp = Slerp([0.0, 1.0], R.from_matrix(rots))
    rot = slerp([alpha]).as_matrix()[0]
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rot
    c2w[:3, 3] = trans
    return torch.from_numpy(c2w).to(device)


def sample_c2w_at_normed_time(poses: torch.Tensor, normed_time: float) -> torch.Tensor:
    """Sample a c2w pose along a trajectory at normalized time in [0, 1]."""
    if len(poses) <= 1:
        return poses[0]
    t = max(0.0, min(1.0, float(normed_time)))
    float_idx = t * (len(poses) - 1)
    frame_lo = int(float_idx)
    frame_hi = min(frame_lo + 1, len(poses) - 1)
    alpha = float_idx - frame_lo
    return interp_c2w_pair(poses[frame_lo], poses[frame_hi], alpha)