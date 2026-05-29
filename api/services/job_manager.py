from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    type: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobManager:
    def __init__(self, jobs_dir: Path) -> None:
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _persist(self, job: Job) -> None:
        job.updated_at = datetime.now(timezone.utc).isoformat()
        self._job_path(job.id).write_text(
            json.dumps(job.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def create(self, job_type: str) -> Job:
        job = Job(id=str(uuid.uuid4()), type=job_type)
        with self._lock:
            self._jobs[job.id] = job
            self._persist(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id]
        path = self._job_path(job_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        job = Job(
            id=data["id"],
            type=data["type"],
            status=JobStatus(data["status"]),
            progress=data.get("progress", 0.0),
            message=data.get("message", ""),
            result=data.get("result", {}),
            error=data.get("error"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def update(
        self,
        job_id: str,
        *,
        status: Optional[JobStatus] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Job:
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"Job not found: {job_id}")
        with self._lock:
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = progress
            if message is not None:
                job.message = message
            if result is not None:
                job.result = {**job.result, **result}
            if error is not None:
                job.error = error
            self._persist(job)
        return job

    def run_async(self, job_id: str, fn: Callable[[Job], None]) -> None:
        def _worker() -> None:
            job = self.get(job_id)
            if job is None:
                return
            try:
                self.update(job_id, status=JobStatus.RUNNING, message="处理中...")
                fn(job)
                self.update(job_id, status=JobStatus.COMPLETED, progress=100.0, message="完成")
            except Exception as exc:  # noqa: BLE001
                self.update(
                    job_id,
                    status=JobStatus.FAILED,
                    error=str(exc),
                    message="失败",
                )

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
