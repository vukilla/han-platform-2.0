# SAM-3D Object Init

We integrate SAM‑3D Objects to estimate object meshes and approximate pose.

## Repo
```
git clone https://github.com/facebookresearch/sam-3d-objects external/sam3d
```

## Usage
```python
from xgen.interaction import estimate_object_pose_sam3d

pose, glb_path = estimate_object_pose_sam3d(
    image_path="frame.png",
    mask_path="mask.png",
    output_dir="outputs/sam3d",
)
```

## Notes
- Requires SAM‑3D checkpoints under `external/sam3d/checkpoints/<tag>/pipeline.yaml`.
- `estimate_object_pose_sam3d` returns centroid‑based pose + exported GLB mesh.
