# LHM-plusplus 补丁记录

本会话中曾对 **LHM-plusplus 仓库** 做过修改；现已 **全部回退**，等价逻辑移至 `api/services/lhm_infer_utils.py`，后续升级 LHM++ 时无需 merge 冲突。

## 改动文件数

| 数量 | 路径 |
| --- | --- |
| **1** | `LHM-plusplus/scripts/inference/to_gs_ply.py` |

（`LHM-plusplus/` 在 `.gitignore` 中，git 不会跟踪该目录的 diff。）

## 补丁内容摘要

1. **新增** `normalize_ref_imgs()` — 多视角参考图 PadRatioWithScale 对齐，修复尺寸不一致导致 `np.concatenate` 失败。
2. **延迟 import** `PoseEstimator` — 仅在 `use_smplx_shape_estimator=true` 时加载，PixelShuffle 路径避免强依赖 pytorch3d。
3. **`setup_loaders_and_inputs`** — 在 `obtain_ref_imgs` 之后按 `max_image_size` 归一化图像。

完整 unified diff 见同目录 [`to_gs_ply.py.patch`](to_gs_ply.py.patch)。

## HRM 侧替代实现

| 原 LHM++ 改动 | HRM 替代 |
| --- | --- |
| `normalize_ref_imgs` | `api/services/lhm_infer_utils.py` |
| lazy `PoseEstimator` + 归一化 + `max_image_size` | `setup_loaders_for_hrm()` 同上 |
| 重建入口 | `lhmpp_service._reconstruct_avatar_real` 调用 `setup_loaders_for_hrm` |
| 动作驱动归一化 | `lhmpp_service._animate_avatar_real` 调用 `normalize_ref_imgs` |

## 可选：手动打补丁到 LHM++

若希望 CLI `python scripts/inference/to_gs_ply.py` 也具备相同行为，可在 LHM++ 根目录执行：

```bash
patch -p1 < /path/to/hrm/api/patches/to_gs_ply.py.patch
```

HRM API **不依赖** 该补丁。
