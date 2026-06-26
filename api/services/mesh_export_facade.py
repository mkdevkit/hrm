"""按配置选择 FBX 导出后端：poisson（默认）| sugar | legacy_smpl。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


def export_fbx_from_gaussian(
    ply_path: Path,
    output_dir: Path,
    *,
    lhm_root: Path | None = None,
    ref_image_dir: Path | None = None,
    betas: Any = None,
) -> dict[str, Any]:
    backend = (settings.mesh_export_backend or "poisson").strip().lower()
    logger.info("FBX 导出后端: %s", backend)

    if backend == "sugar":
        from services.sugar_export_service import export_sugar_fbx_from_gaussian

        return export_sugar_fbx_from_gaussian(
            ply_path,
            output_dir,
            ref_image_dir=ref_image_dir,
        )

    if backend in ("legacy_smpl", "smpl"):
        from services.mesh_export_service import export_skinned_mesh_from_gaussian

        root = str(lhm_root) if lhm_root else settings.lhm_root
        return export_skinned_mesh_from_gaussian(
            ply_path,
            output_dir,
            betas=betas,
            lhm_root=root,
        )

    from services.poisson_mesh_export_service import export_poisson_fbx_from_gaussian

    return export_poisson_fbx_from_gaussian(ply_path, output_dir, lhm_root=lhm_root)
