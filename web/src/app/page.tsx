"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  animateAvatar,
  createAvatar,
  getHealth,
  jobVideoUrl,
  modelUrl,
  skinnedMeshUrl,
  skeletonUrl,
  pollJob,
  previewUrl,
  wsMotionUrl,
  type Job,
} from "@/lib/api";

type Step = "upload" | "reconstructing" | "ready" | "animating";

export default function HomePage() {
  const [images, setImages] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [refView, setRefView] = useState(8);
  const [exportSkinnedMesh, setExportSkinnedMesh] = useState(false);
  const [hasSkinnedMesh, setHasSkinnedMesh] = useState(false);
  const [hasFbx, setHasFbx] = useState(false);
  const [avatarId, setAvatarId] = useState<string | null>(null);
  const [step, setStep] = useState<Step>("upload");
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mockMode, setMockMode] = useState(false);
  const [motionVideo, setMotionVideo] = useState<File | null>(null);
  const [motionFrames, setMotionFrames] = useState(120);
  const [resultVideo, setResultVideo] = useState<string | null>(null);

  // 摄像头流
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [bufferCount, setBufferCount] = useState(0);
  const [streamJobId, setStreamJobId] = useState<string | null>(null);

  useEffect(() => {
    getHealth().then((h) => setMockMode(!!h.mock_mode)).catch(() => {});
  }, []);

  const onFiles = useCallback((files: FileList | null) => {
    if (!files?.length) return;
    const list = Array.from(files).slice(0, 8);
    setImages(list);
    setPreviews(list.map((f) => URL.createObjectURL(f)));
    setError(null);
  }, []);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    onFiles(e.dataTransfer.files);
  };

  const startReconstruct = async () => {
    if (!images.length) {
      setError("请上传至少一张人物图片");
      return;
    }
    setError(null);
    setStep("reconstructing");
    try {
      const { avatar_id, job_id } = await createAvatar(images, refView, exportSkinnedMesh);
      setAvatarId(avatar_id);
      const finalJob = await pollJob(job_id, setJob);
      setHasSkinnedMesh(exportSkinnedMesh && !!finalJob.result?.skeleton_json_path);
      setHasFbx(exportSkinnedMesh && !!finalJob.result?.mesh_fbx_path);
      setStep("ready");
    } catch (e) {
      setError(e instanceof Error ? e.message : "重建失败");
      setStep("upload");
    }
  };

  const startAnimate = async () => {
    if (!avatarId || !motionVideo) {
      setError("请上传动作视频");
      return;
    }
    setError(null);
    setStep("animating");
    setResultVideo(null);
    try {
      const { job_id } = await animateAvatar(avatarId, motionVideo, motionFrames);
      await pollJob(job_id, setJob);
      setResultVideo(jobVideoUrl(job_id));
      setStep("ready");
    } catch (e) {
      setError(e instanceof Error ? e.message : "动画生成失败");
      setStep("ready");
    }
  };

  const startWebcam = async () => {
    if (!avatarId) return;
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const ws = new WebSocket(wsMotionUrl(avatarId));
      wsRef.current = ws;

      ws.onopen = () => {
        setStreaming(true);
        setBufferCount(0);
        captureTimerRef.current = setInterval(() => {
          const video = videoRef.current;
          const canvas = canvasRef.current;
          if (!video || !canvas || ws.readyState !== WebSocket.OPEN) return;
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const ctx = canvas.getContext("2d");
          if (!ctx) return;
          ctx.drawImage(video, 0, 0);
          const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
          ws.send(JSON.stringify({ type: "frame", data: dataUrl }));
        }, 100);
      };

      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === "buffer") setBufferCount(msg.count);
        if (msg.type === "job_started") setStreamJobId(msg.job_id);
        if (msg.type === "job_status") {
          if (msg.status === "completed" && msg.video_url) {
            setResultVideo(`${process.env.NEXT_PUBLIC_API_URL || ""}${msg.video_url}`);
          }
          if (msg.status === "failed") setError(msg.error || "流式动画失败");
        }
        if (msg.type === "error") setError(msg.message);
      };

      ws.onerror = () => setError("WebSocket 连接失败");
    } catch (e) {
      setError(e instanceof Error ? e.message : "无法访问摄像头");
    }
  };

  const stopWebcam = () => {
    if (captureTimerRef.current) clearInterval(captureTimerRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    wsRef.current?.close();
    setStreaming(false);
  };

  const flushStream = () => {
    wsRef.current?.send(JSON.stringify({ type: "flush" }));
  };

  const pollStreamJob = () => {
    if (!streamJobId) return;
    wsRef.current?.send(JSON.stringify({ type: "poll", job_id: streamJobId }));
  };

  useEffect(() => {
    if (!streamJobId || !streaming) return;
    const t = setInterval(pollStreamJob, 2000);
    return () => clearInterval(t);
  }, [streamJobId, streaming]);

  useEffect(() => () => stopWebcam(), []);

  return (
    <main className="container">
      <header className="hero">
        <h1>LHM++ 人体 3D 重建与动作驱动</h1>
        <p>
          上传多视角人物图片生成可动画 3D 模型，再通过动作视频或摄像头实时捕获驱动角色。
          基于{" "}
          <a href="https://lingtengqiu.github.io/LHM++/" target="_blank" rel="noreferrer">
            LHM++
          </a>
          。
        </p>
        {mockMode && (
          <p className="hint" style={{ color: "#fbbf24" }}>
            当前为 Mock 模式（无 GPU），仅用于前端联调。
          </p>
        )}
      </header>

      {/* Step 1: 上传图片 */}
      <section className="card">
        <h2>1. 上传参考图片（1–8 张，全身人物）</h2>
        <div
          className="dropzone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          <p>拖拽图片到此处，或点击选择</p>
          <input
            type="file"
            accept="image/*"
            multiple
            style={{ marginTop: "1rem" }}
            onChange={(e) => onFiles(e.target.files)}
          />
        </div>
        {previews.length > 0 && (
          <div className="thumb-grid">
            {previews.map((src, i) => (
              <img key={i} src={src} alt={`参考图 ${i + 1}`} />
            ))}
          </div>
        )}
        <div className="field" style={{ marginTop: "1rem" }}>
          <label>参考视角数量: {refView}</label>
          <input
            type="range"
            min={1}
            max={8}
            value={refView}
            onChange={(e) => setRefView(Number(e.target.value))}
          />
        </div>
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.6rem",
            marginTop: "1rem",
            cursor: "pointer",
            fontSize: "0.95rem",
          }}
        >
          <input
            type="checkbox"
            checked={exportSkinnedMesh}
            onChange={(e) => setExportSkinnedMesh(e.target.checked)}
            style={{ width: 18, height: 18, accentColor: "var(--accent)" }}
          />
          同时导出 SMPL-X 蒙皮网格（FBX + 骨骼 JSON，含 55 关节 LBS）
        </label>
        <p className="hint">
          开启后除 Gaussian PLY 外，还会生成 Unity/Maya 可用的蒙皮 FBX 与骨骼 JSON（需服务器安装 Blender）。
        </p>
        {step === "upload" && (
          <button className="btn" onClick={startReconstruct} disabled={!images.length}>
            开始 3D 重建
          </button>
        )}
      </section>

      {/* 重建进度 */}
      {step === "reconstructing" && job && (
        <section className="card">
          <h2>正在重建 3D 模型</h2>
          <span className={`status-badge ${job.status}`}>{job.status}</span>
          <div className="progress-bar">
            <div style={{ width: `${job.progress}%` }} />
          </div>
          <p className="hint">{job.message}</p>
        </section>
      )}

      {/* Step 2: 3D 结果 */}
      {avatarId && step !== "upload" && step !== "reconstructing" && (
        <section className="card">
          <h2>2. 3D 模型</h2>
          <div className="grid-2">
            <div>
              <img
                src={previewUrl(avatarId)}
                alt="预览"
                style={{ width: "100%", borderRadius: 12, border: "1px solid var(--border)" }}
              />
            </div>
            <div>
              <p className="hint">Avatar ID: {avatarId}</p>
              <a className="btn btn-secondary" href={modelUrl(avatarId)} download style={{ marginTop: "1rem" }}>
                下载 Gaussian Splat PLY
              </a>
              {hasSkinnedMesh && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginTop: "0.75rem" }}>
                  {hasFbx && (
                    <a className="btn btn-secondary" href={skinnedMeshUrl(avatarId, "fbx")} download>
                      下载蒙皮 FBX
                    </a>
                  )}
                  <a className="btn btn-secondary" href={skeletonUrl(avatarId)} download>
                    下载 SMPL-X 骨骼 JSON
                  </a>
                </div>
              )}
              <p className="hint" style={{ marginTop: "0.75rem" }}>
                PLY 请用 <a href="http://localhost:5174" target="_blank" rel="noreferrer">3DGS 查看器</a>（<code>cd 3dgs && npm run dev</code>）预览；FBX 可直接导入 Unity / Maya。
                {hasSkinnedMesh && !hasFbx && " FBX 未生成，请确认 API 服务器已安装 Blender。"}
              </p>
            </div>
          </div>
        </section>
      )}

      {/* Step 3: 动作驱动 */}
      {avatarId && (step === "ready" || step === "animating") && (
        <section className="card">
          <h2>3. 动作捕获与驱动</h2>

          <div className="grid-2">
            <div>
              <h3 style={{ fontSize: "0.95rem", marginBottom: "0.75rem" }}>方式 A：上传动作视频</h3>
              <input
                type="file"
                accept="video/*"
                onChange={(e) => setMotionVideo(e.target.files?.[0] || null)}
              />
              <div className="field" style={{ marginTop: "1rem" }}>
                <label>渲染帧数: {motionFrames}</label>
                <input
                  type="range"
                  min={30}
                  max={300}
                  step={2}
                  value={motionFrames}
                  onChange={(e) => setMotionFrames(Number(e.target.value))}
                />
              </div>
              <button
                className="btn"
                onClick={startAnimate}
                disabled={!motionVideo || step === "animating"}
                style={{ marginTop: "0.5rem" }}
              >
                {step === "animating" ? "渲染中..." : "生成动画"}
              </button>
            </div>

            <div>
              <h3 style={{ fontSize: "0.95rem", marginBottom: "0.75rem" }}>方式 B：摄像头实时捕获</h3>
              <video ref={videoRef} className="video-preview" muted playsInline style={{ maxHeight: 240 }} />
              <canvas ref={canvasRef} style={{ display: "none" }} />
              <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.75rem", flexWrap: "wrap" }}>
                {!streaming ? (
                  <button className="btn btn-secondary" onClick={startWebcam}>
                    开启摄像头
                  </button>
                ) : (
                  <>
                    <button className="btn" onClick={flushStream}>
                      提交动作 ({bufferCount} 帧)
                    </button>
                    <button className="btn btn-secondary" onClick={stopWebcam}>
                      停止
                    </button>
                  </>
                )}
              </div>
              <p className="hint">至少采集 30 帧后点击「提交动作」进行推理。</p>
            </div>
          </div>

          {step === "animating" && job && (
            <div style={{ marginTop: "1rem" }}>
              <span className={`status-badge ${job.status}`}>{job.status}</span>
              <div className="progress-bar">
                <div style={{ width: `${job.progress}%` }} />
              </div>
              <p className="hint">{job.message}</p>
            </div>
          )}
        </section>
      )}

      {/* 动画结果 */}
      {resultVideo && (
        <section className="card">
          <h2>动画结果</h2>
          <video src={resultVideo} className="video-preview" controls autoPlay loop />
        </section>
      )}

      {error && <p className="error">{error}</p>}
    </main>
  );
}
