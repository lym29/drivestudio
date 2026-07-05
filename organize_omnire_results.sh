#!/bin/bash
# Organize existing OmniRe reconstruction results into a unified directory.

set -euo pipefail

SCENE_IDS=("000" "001" "002" "003" "004" "005" "006" "008" "009")
SOURCE_ROOT="${SOURCE_ROOT:-/DATA/OmniRe-output/recon}"
SPLIT_ROOT="${SPLIT_ROOT:-/DATA/OmniRe-output/split_videos}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/DATA/lym_data/omnire-results}"
PROJECT_NAME="${PROJECT_NAME:-exp-nusences-mini}"

VIDEO_SUFFIXES=(
    "gt_rgbs:input_multiview_gt.mp4"
    "Background_rgbs:background_render.mp4"
    "RigidNodes_rgbs:dynamic_vehicles_rigidnodes.mp4"
    "Dynamic_rgbs:dynamic_all.mp4"
    "rgbs:full_render.mp4"
)

find_best_step() {
    local scene_dir="$1"
    local best_step=""
    local search_dirs=("$scene_dir/videos" "$scene_dir/videos_eval")

    for dir in "${search_dirs[@]}"; do
        [ -d "$dir" ] || continue
        for file in "$dir"/full_set_*_gt_rgbs.mp4; do
            [ -f "$file" ] || continue
            local step
            step=$(basename "$file" | sed -E 's/full_set_([0-9]+)_gt_rgbs\.mp4/\1/')
            if [ -z "$best_step" ] || [ "$step" -gt "$best_step" ]; then
                best_step="$step"
            fi
        done
    done

    echo "$best_step"
}

find_video_file() {
    local scene_dir="$1"
    local step="$2"
    local suffix="$3"
    local candidates=(
        "$scene_dir/videos/full_set_${step}_${suffix}.mp4"
        "$scene_dir/videos_eval/full_set_${step}_${suffix}.mp4"
    )

    for candidate in "${candidates[@]}"; do
        if [ -f "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

link_per_camera_videos() {
    local scene_id="$1"
    local step="$2"
    local per_camera_dir="$3"
    local split_dir="$SPLIT_ROOT/$scene_id"

    mkdir -p "$per_camera_dir"

    if [ -d "$split_dir" ] && [ "$(ls -A "$split_dir" 2>/dev/null | wc -l)" -gt 0 ]; then
        for src in "$split_dir"/full_set_*_gt_rgbs_CAM_*.mp4; do
            [ -f "$src" ] || continue
            local cam_name
            cam_name=$(basename "$src" | sed -E 's/.*_(CAM_[A-Z_]+)\.mp4/\1/')
            ln -sfn "../../../../../OmniRe-output/split_videos/${scene_id}/$(basename "$src")" \
                "$per_camera_dir/${cam_name}.mp4"
        done
        echo "linked_existing_split"
        return 0
    fi

    local gt_video
    if gt_video=$(find_video_file "$SOURCE_ROOT/${PROJECT_NAME}-${scene_id}" "$step" "gt_rgbs"); then
        python tools/split_nuscenes_video.py "$gt_video" "$per_camera_dir/_tmp_split"
        for src in "$per_camera_dir/_tmp_split"/full_set_*_gt_rgbs_CAM_*.mp4; do
            [ -f "$src" ] || continue
            local cam_name
            cam_name=$(basename "$src" | sed -E 's/.*_(CAM_[A-Z_]+)\.mp4/\1/')
            mv "$src" "$per_camera_dir/${cam_name}.mp4"
        done
        rmdir "$per_camera_dir/_tmp_split" 2>/dev/null || true
        echo "generated_from_gt"
        return 0
    fi

    echo "missing"
    return 1
}

mkdir -p "$OUTPUT_ROOT"
manifest_scenes="["

for idx in "${!SCENE_IDS[@]}"; do
    scene_id="${SCENE_IDS[$idx]}"
    scene_name="scene_${scene_id}"
    source_dir="$SOURCE_ROOT/${PROJECT_NAME}-${scene_id}"
    target_dir="$OUTPUT_ROOT/$scene_name"

    echo ""
    echo "--- Organizing scene ${scene_id} ---"

    if [ ! -d "$source_dir" ]; then
        echo "⊘ Skip: source directory not found: $source_dir"
        continue
    fi

    step=$(find_best_step "$source_dir")
    if [ -z "$step" ]; then
        echo "⊘ Skip: no gt video found for scene ${scene_id}"
        continue
    fi

    mkdir -p "$target_dir/models" "$target_dir/videos/per_camera"

    ln -sfn "../../../OmniRe-output/recon/${PROJECT_NAME}-${scene_id}/checkpoint_final.pth" \
        "$target_dir/checkpoint_final.pth"
    ln -sfn "../../../OmniRe-output/recon/${PROJECT_NAME}-${scene_id}/config.yaml" \
        "$target_dir/config.yaml"

    copied_json="["
    missing_json="["
    copied_count=0
    missing_count=0

    for mapping in "${VIDEO_SUFFIXES[@]}"; do
        suffix="${mapping%%:*}"
        target_name="${mapping##*:}"
        if src=$(find_video_file "$source_dir" "$step" "$suffix"); then
            cp -f "$src" "$target_dir/videos/$target_name"
            if [ "$copied_count" -gt 0 ]; then copied_json+=","; fi
            copied_json+="\"$target_name\""
            copied_count=$((copied_count + 1))
        else
            if [ "$missing_count" -gt 0 ]; then missing_json+=","; fi
            missing_json+="\"$target_name\""
            missing_count=$((missing_count + 1))
        fi
    done
    copied_json+="]"
    missing_json+="]"

    per_camera_status=$(link_per_camera_videos "$scene_id" "$step" "$target_dir/videos/per_camera" || true)
    per_camera_count=$(find "$target_dir/videos/per_camera" -maxdepth 1 -name 'CAM_*.mp4' | wc -l)
    checkpoint_size=$(stat -c%s "$source_dir/checkpoint_final.pth")

    if [ "$idx" -gt 0 ]; then
        manifest_scenes+=","
    fi
    manifest_scenes+=$(cat <<EOF

  {
    "scene_id": "${scene_id}",
    "scene_dir": "${target_dir}",
    "source_dir": "${source_dir}",
    "step": ${step},
    "checkpoint_size_bytes": ${checkpoint_size},
    "videos": {
      "copied": ${copied_json},
      "missing": ${missing_json}
    },
    "per_camera": {
      "status": "${per_camera_status}",
      "count": ${per_camera_count}
    }
  }
EOF
)

    echo "✓ Scene ${scene_id}: step=${step}, videos=${copied_count}, per_camera=${per_camera_count}"
done

manifest_scenes+="
]"
manifest_file="$OUTPUT_ROOT/manifest.json"
cat > "$manifest_file" <<EOF
{
  "generated_at": "$(date -Iseconds)",
  "project": "${PROJECT_NAME}",
  "source_root": "${SOURCE_ROOT}",
  "output_root": "${OUTPUT_ROOT}",
  "skipped_scenes": ["007"],
  "scenes": ${manifest_scenes}
}
EOF

echo ""
echo "========================================"
echo "Organization complete"
echo "Output: $OUTPUT_ROOT"
echo "Manifest: $manifest_file"
echo "========================================"
