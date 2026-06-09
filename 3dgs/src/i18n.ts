export type Locale = "zh" | "en";

const STORAGE_KEY = "hrm-3dgs-viewer-locale";

const messages = {
  zh: {
    pageTitle: "HRM 3DGS 查看器",
    appTitle: "HRM 3DGS 查看器",
    openPly: "打开 PLY",
    loadFromUrl: "从 URL 加载",
    statusIdle: "拖放 .ply 到窗口，或点击「打开 PLY」",
    statusLoading: "加载中: {name}",
    statusLoaded: "已加载: {name}",
    statusFailed: "加载失败: {msg}",
    statusDropInvalid: "请拖放 .ply 或 .splat 文件",
    promptUrlTitle: "输入 PLY 文件 URL",
    promptUrlDefault: "http://localhost:8000/api/v1/avatars/{avatar_id}/model",
    helpTitle: "用法",
    helpDrop: "拖放 LHM++ / HRM 导出的 avatar.ply",
    helpUrlParam: "URL 参数：?ply=文件地址",
    helpControls: "鼠标左键旋转 · 右键平移 · 滚轮缩放",
    dragHint: "释放以加载 PLY",
    langSwitch: "EN",
    langSwitchTitle: "切换为英文",
  },
  en: {
    pageTitle: "3DGS Viewer",
    appTitle: "3DGS Viewer",
    openPly: "Open PLY",
    loadFromUrl: "Load from URL",
    statusIdle: "Drop a .ply file here, or click Open PLY",
    statusLoading: "Loading: {name}",
    statusLoaded: "Loaded: {name}",
    statusFailed: "Load failed: {msg}",
    statusDropInvalid: "Please drop a .ply or .splat file",
    promptUrlTitle: "Enter PLY file URL",
    promptUrlDefault: "http://localhost:8000/api/v1/avatars/{avatar_id}/model",
    helpTitle: "Usage",
    helpDrop: "Drop avatar.ply exported from LHM++ / HRM",
    helpUrlParam: "URL param: ?ply=file_url",
    helpControls: "Left drag: rotate · Right drag: pan · Wheel: zoom",
    dragHint: "Release to load PLY",
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
  if (lang.startsWith("zh")) return "zh";
  return "en";
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
  const locale = currentLocale;
  document.documentElement.lang = locale === "zh" ? "zh-CN" : "en";
  document.title = t("pageTitle");

  const map: Record<string, MessageKey> = {
    "app-title": "appTitle",
    "btn-open-ply-label": "openPly",
    "btn-load-url": "loadFromUrl",
    "help-title": "helpTitle",
    "help-drop": "helpDrop",
    "help-url": "helpUrlParam",
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

  const langBtn = document.getElementById("btn-lang");
  langBtn?.addEventListener("click", () => {
    toggleLocale();
    onLocaleChange?.();
  });
}
