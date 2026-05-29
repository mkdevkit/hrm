const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export type JobStatus = "pending" | "running" | "completed" | "failed";

export interface Job {
  id: string;
  type: string;
  status: JobStatus;
  progress: number;
  message: string;
  result: Record<string, unknown>;
  error?: string;
}

export interface AvatarMeta {
  id: string;
  status: string;
  ref_view?: number;
  image_count?: number;
  ply_path?: string;
  preview_path?: string;
  job_id?: string;
  export_skinned_mesh?: boolean;
  mesh_obj_path?: string;
  mesh_glb_path?: string;
  skeleton_json_path?: string;
  joint_count?: number;
  last_animation?: Record<string, unknown>;
}

export async function getHealth() {
  const res = await fetch(`${API_BASE}/api/v1/health`);
  return res.json();
}

export async function createAvatar(
  images: File[],
  refView = 8,
  exportSkinnedMesh = false,
) {
  const form = new FormData();
  images.forEach((img) => form.append("images", img));
  form.append("ref_view", String(refView));
  form.append("export_skinned_mesh", exportSkinnedMesh ? "true" : "false");

  const res = await fetch(`${API_BASE}/api/v1/avatars`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "创建 Avatar 失败");
  }
  return res.json() as Promise<{ avatar_id: string; job_id: string }>;
}

export async function getAvatar(avatarId: string): Promise<AvatarMeta> {
  const res = await fetch(`${API_BASE}/api/v1/avatars/${avatarId}`);
  if (!res.ok) throw new Error("获取 Avatar 失败");
  return res.json();
}

export async function getJob(jobId: string): Promise<Job> {
  const res = await fetch(`${API_BASE}/api/v1/jobs/${jobId}`);
  if (!res.ok) throw new Error("获取任务失败");
  return res.json();
}

export async function animateAvatar(
  avatarId: string,
  motionVideo: File,
  motionFrames = 120,
) {
  const form = new FormData();
  form.append("motion_video", motionVideo);
  form.append("motion_frames", String(motionFrames));

  const res = await fetch(`${API_BASE}/api/v1/avatars/${avatarId}/animate`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "动画生成失败");
  }
  return res.json() as Promise<{ job_id: string }>;
}

export function previewUrl(avatarId: string) {
  return `${API_BASE}/api/v1/avatars/${avatarId}/preview`;
}

export function modelUrl(avatarId: string) {
  return `${API_BASE}/api/v1/avatars/${avatarId}/model`;
}

export function skinnedMeshUrl(avatarId: string, format: "obj" | "glb" = "obj") {
  return `${API_BASE}/api/v1/avatars/${avatarId}/mesh?format=${format}`;
}

export function skeletonUrl(avatarId: string) {
  return `${API_BASE}/api/v1/avatars/${avatarId}/skeleton`;
}

export function jobVideoUrl(jobId: string) {
  return `${API_BASE}/api/v1/jobs/${jobId}/video`;
}

export function wsMotionUrl(avatarId: string) {
  const base = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
  return `${base}/api/v1/avatars/${avatarId}/motion-stream`;
}

export async function pollJob(
  jobId: string,
  onUpdate?: (job: Job) => void,
  intervalMs = 1500,
): Promise<Job> {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const job = await getJob(jobId);
        onUpdate?.(job);
        if (job.status === "completed") {
          resolve(job);
        } else if (job.status === "failed") {
          reject(new Error(job.error || "任务失败"));
        } else {
          setTimeout(tick, intervalMs);
        }
      } catch (e) {
        reject(e);
      }
    };
    tick();
  });
}
