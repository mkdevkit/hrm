export type Locale = "zh" | "en";

const STORAGE_KEY = "hrm-3d-viewer-locale";

const messages = {
  zh: {
    pageTitle: "3D 模型查看器",
    appTitle: "3D 模型查看器",
    openFile: "打开模型",
    loadFromUrl: "从 URL 加载",
    animation: "动作",
    play: "播放",
    pause: "暂停",
    animNone: "（无动作）",
    statusIdle: "拖放 FBX / GLB / OBJ，或点击「打开模型」",
    statusLoading: "加载中: {name}",
    statusLoaded: "已加载: {name}",
    statusLoadedWithAnim: "已加载: {name}（{count} 个动作）",
    statusFailed: "加载失败: {msg}",
    statusDropInvalid: "请拖放 .fbx / .glb / .gltf / .obj 文件",
    promptUrlTitle: "输入模型 URL（FBX / GLB / GLTF / OBJ）",
    helpTitle: "用法",
    helpFormats: "支持 FBX、GLB/GLTF、OBJ",
    helpAnim: "若模型含动作片段，可在「动作」菜单中选择",
    helpHrmNote: "HRM 烘焙贴图按 3DGS 无光照显示；无贴图时用肤色 PBR",
    helpControls: "鼠标左键旋转 · 右键平移 · 滚轮缩放",
    dragHint: "释放以加载模型",
    langSwitch: "EN",
    langSwitchTitle: "切换为英文",
  },
  en: {
    pageTitle: "3D Model Viewer",
    appTitle: "3D Model Viewer",
    openFile: "Open model",
    loadFromUrl: "Load from URL",
    animation: "Animation",
    play: "Play",
    pause: "Pause",
    animNone: "(no animation)",
    statusIdle: "Drop FBX / GLB / OBJ here, or click Open model",
    statusLoading: "Loading: {name}",
    statusLoaded: "Loaded: {name}",
    statusLoadedWithAnim: "Loaded: {name} ({count} clip(s))",
    statusFailed: "Load failed: {msg}",
    statusDropInvalid: "Please drop .fbx / .glb / .gltf / .obj file",
    promptUrlTitle: "Enter model URL (FBX / GLB / GLTF / OBJ)",
    helpTitle: "Usage",
    helpFormats: "Supports FBX, GLB/GLTF, OBJ",
    helpAnim: "If the model has clips, pick one from the Animation menu",
    helpHrmNote: "HRM baked textures shown unlit (like 3DGS); untextured FBX uses skin PBR",
    helpControls: "Left drag: rotate · Right drag: pan · Wheel: zoom",
    dragHint: "Release to load model",
    langSwitch: "中文",
    langSwitchTitle: "Switch to Chinese",
  },
} as const;

export type MessageKey = keyof typeof messages.en;

let currentLocale: Locale = detectSystemLocale();

function detectSystemLocale(): Locale {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "zh" || stored === "en") return stored;
  const lang = (navigator.language || "en").toLowerCase();
  return lang.startsWith("zh") ? "zh" : "en";
}

export function getLocale(): Locale {
  return currentLocale;
}

export function setLocale(locale: Locale): void {
  currentLocale = locale;
  localStorage.setItem(STORAGE_KEY, locale);
  applyLocaleToDocument();
}

export function toggleLocale(): Locale {
  setLocale(currentLocale === "zh" ? "en" : "zh");
  return currentLocale;
}

export function t(key: MessageKey, vars?: Record<string, string>): string {
  let text: string = messages[currentLocale][key];
  if (vars) {
    for (const [k, v] of Object.entries(vars)) {
      text = text.replace(`{${k}}`, v);
    }
  }
  return text;
}

export function applyLocaleToDocument(): void {
  document.documentElement.lang = currentLocale === "zh" ? "zh-CN" : "en";
  document.title = t("pageTitle");

  const map: Record<string, MessageKey> = {
    "app-title": "appTitle",
    "btn-open-file-label": "openFile",
    "btn-load-url": "loadFromUrl",
    "anim-label": "animation",
    "help-title": "helpTitle",
    "help-formats": "helpFormats",
    "help-anim": "helpAnim",
    "help-hrm-note": "helpHrmNote",
    "help-controls": "helpControls",
  };

  for (const [id, key] of Object.entries(map)) {
    const el = document.getElementById(id);
    if (el) el.textContent = t(key);
  }

  const langBtn = document.getElementById("btn-lang");
  if (langBtn) {
    langBtn.textContent = t("langSwitch");
    langBtn.setAttribute("title", t("langSwitchTitle"));
    langBtn.setAttribute("aria-label", t("langSwitchTitle"));
  }

  document.documentElement.style.setProperty("--i18n-drag-hint", `"${t("dragHint")}"`);
}

export function initI18n(onLocaleChange?: () => void): void {
  applyLocaleToDocument();
  document.getElementById("btn-lang")?.addEventListener("click", () => {
    toggleLocale();
    onLocaleChange?.();
  });
}
