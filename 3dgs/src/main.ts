import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";

const host = document.getElementById("canvas-host")!;
const statusEl = document.getElementById("status")!;
const fileInput = document.getElementById("file-input") as HTMLInputElement;
const btnLoadUrl = document.getElementById("btn-load-url")!;

let viewer: GaussianSplats3D.Viewer | null = null;
let loading = false;

const DEFAULT_SCENE_OPTIONS = {
  splatAlphaRemovalThreshold: 1,
  showLoadingUI: true,
  position: [0, 0, 0] as [number, number, number],
  rotation: [0, 0, 0, 1] as [number, number, number, number],
  scale: [1, 1, 1] as [number, number, number],
  progressiveLoad: true,
};

function setStatus(text: string) {
  statusEl.textContent = text;
}

function resolveSceneFormat(source: string, fileName?: string): number {
  const fromName = fileName
    ? GaussianSplats3D.LoaderUtils.sceneFormatFromPath(fileName)
    : null;
  if (fromName != null) return fromName;

  const fromUrl = GaussianSplats3D.LoaderUtils.sceneFormatFromPath(source);
  if (fromUrl != null) return fromUrl;

  return GaussianSplats3D.SceneFormat.Ply;
}

async function disposeViewer() {
  if (viewer) {
    try {
      viewer.dispose();
    } catch {
      /* ignore */
    }
    viewer = null;
  }
  host.innerHTML = "";
}

async function createViewer() {
  await disposeViewer();
  viewer = new GaussianSplats3D.Viewer({
    rootElement: host,
    cameraUp: [0, 1, 0],
    initialCameraPosition: [0, 1.0, 2.5],
    initialCameraLookAt: [0, 0.9, 0],
    sharedMemoryForWorkers: false,
    gpuAcceleratedSort: false,
    dynamicScene: false,
    antialiased: true,
    focalAdjustment: 1.0,
    logLevel: GaussianSplats3D.LogLevel.None,
  });
  return viewer;
}

async function loadFromUrl(url: string, label?: string, fileName?: string) {
  if (loading) return;
  loading = true;
  setStatus(`加载中: ${label ?? url}`);

  try {
    const v = await createViewer();
    const format = resolveSceneFormat(url, fileName ?? label);
    await v.addSplatScene(url, {
      ...DEFAULT_SCENE_OPTIONS,
      format,
    });
    v.start();
    setStatus(`已加载: ${label ?? url}`);
  } catch (err) {
    await disposeViewer();
    const msg = err instanceof Error ? err.message : String(err);
    setStatus(`加载失败: ${msg}`);
    console.error(err);
  } finally {
    loading = false;
  }
}

async function loadFromFile(file: File) {
  const blobUrl = URL.createObjectURL(file);
  try {
    await loadFromUrl(blobUrl, file.name, file.name);
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}

function promptUrl() {
  const example =
    "http://localhost:8000/api/v1/avatars/{avatar_id}/model";
  const input = window.prompt("输入 PLY 文件 URL", example);
  if (input?.trim()) {
    void loadFromUrl(input.trim());
  }
}

function loadFromQuery() {
  const params = new URLSearchParams(window.location.search);
  const ply = params.get("ply");
  if (ply) {
    void loadFromUrl(ply);
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) void loadFromFile(file);
  fileInput.value = "";
});

btnLoadUrl.addEventListener("click", promptUrl);

window.addEventListener("dragover", (e) => {
  e.preventDefault();
  document.body.classList.add("drag-over");
});

window.addEventListener("dragleave", (e) => {
  if (e.relatedTarget === null) {
    document.body.classList.remove("drag-over");
  }
});

window.addEventListener("drop", (e) => {
  e.preventDefault();
  document.body.classList.remove("drag-over");
  const file = e.dataTransfer?.files?.[0];
  if (file && (file.name.endsWith(".ply") || file.name.endsWith(".splat"))) {
    void loadFromFile(file);
  } else {
    setStatus("请拖放 .ply 或 .splat 文件");
  }
});

loadFromQuery();
