import { initI18n, t } from "./i18n";
import { ModelViewer } from "./viewer";

const host = document.getElementById("canvas-host")!;
const statusEl = document.getElementById("status")!;
const fileInput = document.getElementById("file-input") as HTMLInputElement;
const btnLoadUrl = document.getElementById("btn-load-url")!;
const animWrap = document.getElementById("anim-wrap")!;
const animSelect = document.getElementById("anim-select") as HTMLSelectElement;
const btnPlay = document.getElementById("btn-play")!;

const ACCEPT_EXT = [".fbx", ".glb", ".gltf", ".obj"];

let viewer: ModelViewer | null = null;
let loading = false;
let lastStatusKey: "idle" | "custom" = "idle";
let lastCustomStatus = "";

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
  if (lastStatusKey === "idle") setStatusIdle();
  else statusEl.textContent = lastCustomStatus;
  updatePlayButtonLabel(viewer?.isPlaying() ?? true);
}

function updatePlayButtonLabel(playing: boolean) {
  btnPlay.textContent = playing ? t("pause") : t("play");
  btnPlay.setAttribute("aria-pressed", String(playing));
}

function updateAnimationUi(clips: { name: string }[]) {
  animSelect.innerHTML = "";
  if (clips.length === 0) {
    animWrap.classList.add("hidden");
    btnPlay.classList.add("hidden");
    return;
  }

  animWrap.classList.remove("hidden");
  btnPlay.classList.remove("hidden");

  clips.forEach((clip, index) => {
    const opt = document.createElement("option");
    opt.value = String(index);
    opt.textContent = clip.name || `Clip ${index + 1}`;
    animSelect.appendChild(opt);
  });
}

function isAcceptedFile(file: File): boolean {
  const lower = file.name.toLowerCase();
  return ACCEPT_EXT.some((ext) => lower.endsWith(ext));
}

function resolveHrmModelUrl(params: URLSearchParams): string | null {
  const avatarId = params.get("avatar");
  if (!avatarId) return null;

  const apiBase = (params.get("api") || "http://localhost:8000").replace(/\/$/, "");
  const format = (params.get("format") || "fbx").toLowerCase();
  if (format !== "fbx" && format !== "obj") {
    throw new Error("HRM avatar format must be fbx or obj");
  }
  return `${apiBase}/api/v1/avatars/${avatarId}/mesh?format=${format}`;
}

function resolveInitialUrl(): string | null {
  const params = new URLSearchParams(window.location.search);
  const direct =
    params.get("model") ||
    params.get("fbx") ||
    params.get("glb") ||
    params.get("gltf") ||
    params.get("obj");
  if (direct) return direct;
  return resolveHrmModelUrl(params);
}

async function ensureViewer() {
  if (!viewer) {
    viewer = new ModelViewer(host, {
      onAnimationsChange: (clips) => updateAnimationUi(clips),
      onPlayingChange: (playing) => updatePlayButtonLabel(playing),
    });
  }
  return viewer;
}

async function loadSource(source: string, displayName?: string) {
  if (loading) return;
  loading = true;
  setStatusCustom(t("statusLoading", { name: displayName || source }));

  try {
    const v = await ensureViewer();
    const result = await v.loadFromUrl(source, displayName);
    if (result.animations.length > 0) {
      setStatusCustom(
        t("statusLoadedWithAnim", {
          name: result.displayName,
          count: String(result.animations.length),
        }),
      );
    } else {
      setStatusCustom(t("statusLoaded", { name: result.displayName }));
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setStatusCustom(t("statusFailed", { msg }));
  } finally {
    loading = false;
  }
}

async function loadFile(file: File) {
  if (!isAcceptedFile(file)) {
    setStatusCustom(t("statusDropInvalid"));
    return;
  }
  if (loading) return;
  loading = true;
  setStatusCustom(t("statusLoading", { name: file.name }));

  try {
    const v = await ensureViewer();
    const result = await v.loadFromFile(file);
    if (result.animations.length > 0) {
      setStatusCustom(
        t("statusLoadedWithAnim", {
          name: result.displayName,
          count: String(result.animations.length),
        }),
      );
    } else {
      setStatusCustom(t("statusLoaded", { name: result.displayName }));
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setStatusCustom(t("statusFailed", { msg }));
  } finally {
    loading = false;
  }
}

function setupDragDrop() {
  let dragDepth = 0;

  window.addEventListener("dragenter", (e) => {
    e.preventDefault();
    dragDepth += 1;
    document.body.classList.add("drag-over");
  });

  window.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) document.body.classList.remove("drag-over");
  });

  window.addEventListener("dragover", (e) => e.preventDefault());

  window.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    document.body.classList.remove("drag-over");
    const file = e.dataTransfer?.files?.[0];
    if (file) void loadFile(file);
  });
}

function setupControls() {
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) void loadFile(file);
    fileInput.value = "";
  });

  btnLoadUrl.addEventListener("click", () => {
    const url = window.prompt(t("promptUrlTitle"), "");
    if (url?.trim()) void loadSource(url.trim());
  });

  animSelect.addEventListener("change", () => {
    const index = Number.parseInt(animSelect.value, 10);
    viewer?.playClip(index);
  });

  btnPlay.addEventListener("click", () => {
    viewer?.togglePlaying();
  });
}

async function bootstrap() {
  initI18n(refreshStatusOnLocaleChange);
  setStatusIdle();
  setupDragDrop();
  setupControls();

  const initial = resolveInitialUrl();
  if (initial) {
    const params = new URLSearchParams(window.location.search);
    const name =
      params.get("avatar") != null
        ? `avatar.${params.get("format") || "fbx"}`
        : initial.split("/").pop()?.split("?")[0];
    await loadSource(initial, name || undefined);
  }
}

void bootstrap();
