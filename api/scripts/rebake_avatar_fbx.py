#!/usr/bin/env python3
"""对已有 avatar output 重新导出 FBX。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_paths() -> None:
    if str(API_ROOT) not in sys.path:
        sys.path.insert(0, str(API_ROOT))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    _bootstrap_paths()
    parser = argparse.ArgumentParser(description="重新导出 avatar.fbx")
    parser.add_argument("output_dir", type=Path, help="avatar output 目录（含 avatar.ply）")
    parser.add_argument("--ply", type=Path, default=None, help="PLY 路径（默认 output_dir/avatar.ply）")
    parser.add_argument(
        "--backend",
        choices=("poisson", "sugar", "legacy_smpl"),
        default=None,
        help="覆盖 MESH_EXPORT_BACKEND",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    ply = args.ply or (output_dir / "avatar.ply")
    if not ply.is_file():
        print(f"缺少 PLY: {ply}", file=sys.stderr)
        return 1

    if args.backend:
        from config import settings

        settings.mesh_export_backend = args.backend

    from services.mesh_export_facade import export_fbx_from_gaussian

    result = export_fbx_from_gaussian(ply, output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("mesh_fbx_path") else 1


if __name__ == "__main__":
    raise SystemExit(main())
