#!/usr/bin/env python3
"""对已有 avatar output 目录重烘焙 3DGS 贴图并重新导出 FBX。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="重烘焙 avatar_diffuse.png 并导出 avatar_skinned.fbx")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="avatar output 目录（含 avatar_skinned.obj、avatar_lbs_weights.npz）",
    )
    parser.add_argument(
        "--ply",
        type=Path,
        default=None,
        help="3DGS PLY 路径（默认在 output 或上级目录查找 avatar.ply）",
    )
    args = parser.parse_args()

    from services.mesh_export_service import rebake_skinned_mesh_from_output_dir

    result = rebake_skinned_mesh_from_output_dir(args.output_dir, ply_path=args.ply)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("mesh_fbx_path") else 1


if __name__ == "__main__":
    raise SystemExit(main())
