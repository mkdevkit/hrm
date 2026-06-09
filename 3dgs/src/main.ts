import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import { initI18n, t } from "./i18n";

const host = document.getElementById("canvas-host")!;
const statusEl = document.getElementById("status")!;
const fileInput = document.getElementById("file-input") as HTMLInputElement;
const btnLoadUrl = document.getElementById("btn-load-url")!;

let viewer: GaussianSplats3D.Viewer | null = null;
let loading = false;
let lastStatusKey: "idle" | "custom" = "idle";
let lastCustomStatus = "";

const DEFAULT_SCENE_OPTIONS = {
  splatAlphaRemovalThreshold: 1,
  showLoadingUI: true,
  position: [0, 0, 0] as [number, number, number],
  rotation: [0, 0, 0, 1] as [number, number, number, number],
  scale: [1, 1, 1] as [number, number, number],
  progressiveLoad: true,
};

function setStatusIdle() {
  lastStatusKey = "idle";
  statusEl.textContent = t("statusIdle");
}

function setStatusCustom(text: string) {
  lastStatusKey = "custom";
  lastCustomStatus = text;
  statusEl.textContent = text;
}

function refreshStatusOnLocaleChange() {
  if (lastStatusKey === "idle") {
    setStatusIdle();
  } else {
    statusEl.textContent = lastCustomStatus;
  }
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
  const displayName = label ?? url;
  setStatusCustom(t("statusLoading", { name: displayName }));

  try {
    const v = await createViewer();
    const format = resolveSceneFormat(url, fileName ?? label);
    await v.addSplatScene(url, {
      ...DEFAULT_SCENE_OPTIONS,
      format,
    });
    v.start();
    setStatusCustom(t("statusLoaded", { name: displayName }));
  } catch (err) {
    await disposeViewer();
    const msg = err instanceof Error ? err.message : String(err);
    setStatusCustom(t("statusFailed", { msg }));
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
  const input = window.prompt(t("promptUrlTitle"), t("promptUrlDefault"));
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

initI18n(refreshStatusOnLocaleChange);
setStatusIdle();

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
    setStatusCustom(t("statusDropInvalid"));
  }
});

loadFromQuery();
