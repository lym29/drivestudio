# NuScenes Video Splitting Tools

## 脚本说明

### 1. `split_nuscenes_video.py` - 单视频分割工具
基础工具，将一个 NuScenes 6相机拼接视频分割成6个独立视频。

**用法：**
```bash
python tools/split_nuscenes_video.py <input_video> <output_dir>
```

**示例：**
```bash
python tools/split_nuscenes_video.py \
    full_set_40000_rgbs.mp4 \
    output/
```

**输出：**
- `output/full_set_40000_rgbs_CAM_FRONT_LEFT.mp4`
- `output/full_set_40000_rgbs_CAM_FRONT.mp4`
- `output/full_set_40000_rgbs_CAM_FRONT_RIGHT.mp4`
- `output/full_set_40000_rgbs_CAM_BACK_LEFT.mp4`
- `output/full_set_40000_rgbs_CAM_BACK.mp4`
- `output/full_set_40000_rgbs_CAM_BACK_RIGHT.mp4`

---

### 2. `split_scene_videos.sh` - 批量场景视频分割工具
自动处理多个场景的特定视频文件。

**处理的视频：**
- `full_set_40000_Background_rgbs.mp4` (背景渲染)
- `full_set_40000_rgbs.mp4` (完整渲染)
- `full_set_40000_gt_rgbs.mp4` (Ground Truth)

**输入路径：**
```
/DATA/OmniRe-output/recon/exp-nusences-mini-{scene_id}/videos_eval/
```

**输出路径：**
```
/DATA/OmniRe-output/split_videos/{scene_id}/
```

---

## 使用方法

### 方法1：处理所有场景（0-149）

```bash
bash split_scene_videos.sh
```

### 方法2：处理指定场景

```bash
# 处理单个场景
bash split_scene_videos.sh 002

# 处理多个场景
bash split_scene_videos.sh 001 002 003 010

# 处理场景范围
bash split_scene_videos.sh $(seq -f "%03g" 0 10)
```

---

## 输出结构示例

```
/DATA/OmniRe-output/split_videos/
├── 001/
│   ├── full_set_40000_Background_rgbs_CAM_FRONT_LEFT.mp4
│   ├── full_set_40000_Background_rgbs_CAM_FRONT.mp4
│   ├── full_set_40000_Background_rgbs_CAM_FRONT_RIGHT.mp4
│   ├── full_set_40000_Background_rgbs_CAM_BACK_LEFT.mp4
│   ├── full_set_40000_Background_rgbs_CAM_BACK.mp4
│   ├── full_set_40000_Background_rgbs_CAM_BACK_RIGHT.mp4
│   ├── full_set_40000_rgbs_CAM_FRONT_LEFT.mp4
│   ├── full_set_40000_rgbs_CAM_FRONT.mp4
│   ├── ... (6个相机 × 3种视频 = 18个视频文件)
│   └── full_set_40000_gt_rgbs_CAM_BACK_RIGHT.mp4
├── 002/
│   └── ... (同上)
└── ...
```

---

## 注意事项

1. **自动跳过不存在的视频**：如果某个场景的视频文件不存在，会自动跳过并显示提示

2. **目录自动创建**：输出目录会自动创建，无需手动创建

3. **视频命名规则**：
   ```
   {原始文件名}_{相机名}.mp4
   ```

4. **场景ID格式**：使用3位数字格式，如 `001`, `002`, `010`

5. **修改项目名**：如果你的项目名不是 `exp-nusences-mini`，可以修改脚本第6行：
   ```bash
   PROJECT_NAME="your-project-name"
   ```

---

## 示例输出

```bash
$ bash split_scene_videos.sh 001 002

========================================
NuScenes Video Splitting Tool
========================================
Processing 2 scene(s)
Videos to split per scene: 3
========================================

--- Scene: 001 ---
  ✓ Processing: full_set_40000_Background_rgbs.mp4
  ✓ Processing: full_set_40000_rgbs.mp4
  ✓ Processing: full_set_40000_gt_rgbs.mp4
  → Saved to: /DATA/OmniRe-output/split_videos/001/
  → Processed: 3 video(s)

--- Scene: 002 ---
  ✓ Processing: full_set_40000_Background_rgbs.mp4
  ⊘ Skip: Video not found: full_set_40000_rgbs.mp4
  ✓ Processing: full_set_40000_gt_rgbs.mp4
  → Saved to: /DATA/OmniRe-output/split_videos/002/
  → Processed: 2 video(s)

========================================
Summary
========================================
Total scenes checked: 2
Total videos processed: 5
Scenes skipped: 0
Output directory: /DATA/OmniRe-output/split_videos
========================================
✓ Done!
```

