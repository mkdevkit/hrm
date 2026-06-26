#!/usr/bin/env python3
"""CLI：对已有 Avatar 用 SuGaR 全库方案导出 FBX。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_API_ROOT = Path(__file__).resolve().parents[1]
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="SuGaR 全库 → avatar.fbx")
    parser.add_argument("avatar_dir", type=Path, help="Avatar output 目录（含 avatar.ply）")
    args = parser.parse_args()

    avatar_dir = args.avatar_dir.resolve()
    ply = avatar_dir / "avatar.ply"
    if not ply.is_file():
        print(f"缺少 {ply}", file=sys.stderr)
        return 1

    from config import settings

    settings.mesh_export_backend = "sugar"
    from services.mesh_export_facade import export_fbx_from_gaussian

    result = export_fbx_from_gaussian(ply, avatar_dir)
    print(result)
    return 0 if result.get("mesh_fbx_path") else 1


if __name__ == "__main__":
    raise SystemExit(main())
