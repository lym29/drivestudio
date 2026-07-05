"""
Split NuScenes 6-camera tiled video into separate videos for each camera.

NuScenes Layout (2 rows x 3 cols):
    ################################################################
    # CAM_FRONT_LEFT  #     CAM_FRONT      #     CAM_FRONT_RIGHT   #
    ################################################################
    #  CAM_BACK_LEFT  #     CAM_BACK       #     CAM_BACK_RIGHT    #
    ################################################################

Usage:
    python tools/split_nuscenes_video.py input_video.mp4 output_dir/
"""

import sys
import os
import cv2
from tqdm import tqdm


def split_nuscenes_video(input_video, output_dir):
    """Split NuScenes 6-camera video into 6 separate videos."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_video}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate camera regions (2 rows x 3 cols)
    cam_width = width // 3
    cam_height = height // 2
    
    print(f"Input video: {input_video}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    print(f"Each camera: {cam_width}x{cam_height}")
    
    # Define camera names and regions (x, y, width, height)
    cameras = {
        'CAM_FRONT_LEFT':   (0,           0,          cam_width, cam_height),
        'CAM_FRONT':        (cam_width,   0,          cam_width, cam_height),
        'CAM_FRONT_RIGHT':  (2*cam_width, 0,          cam_width, cam_height),
        'CAM_BACK_LEFT':    (0,           cam_height, cam_width, cam_height),
        'CAM_BACK':         (cam_width,   cam_height, cam_width, cam_height),
        'CAM_BACK_RIGHT':   (2*cam_width, cam_height, cam_width, cam_height),
    }
    
    # Create video writers for each camera
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    writers = {}
    
    print("\nCreating output videos:")
    for cam_name, (x, y, w, h) in cameras.items():
        output_path = os.path.join(output_dir, f"{base_name}_{cam_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writers[cam_name] = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        print(f"  {output_path}")
    
    # Process frames
    print(f"\nProcessing {total_frames} frames...")
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop and save each camera view
        for cam_name, (x, y, w, h) in cameras.items():
            cropped = frame[y:y+h, x:x+w]
            writers[cam_name].write(cropped)
    
    # Release resources
    cap.release()
    for writer in writers.values():
        writer.release()
    
    print(f"\n✓ Done! 6 videos saved to {output_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python split_nuscenes_video.py input_video.mp4 output_dir/")
        print("Example: python split_nuscenes_video.py test_set_40000_rgbs.mp4 split_output/")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_dir = sys.argv[2]
    
    split_nuscenes_video(input_video, output_dir)

