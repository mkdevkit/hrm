import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";

export type ModelFormat = "fbx" | "gltf" | "obj";

export interface LoadedModel {
  root: THREE.Object3D;
  animations: THREE.AnimationClip[];
  displayName: string;
}

export interface ModelViewerOptions {
  onAnimationsChange?: (clips: THREE.AnimationClip[]) => void;
  onPlayingChange?: (playing: boolean) => void;
}

function inferFormat(source: string, hint?: string): ModelFormat {
  const name = (hint || source).split(/[?#]/)[0]?.toLowerCase() || "";
  if (name.endsWith(".fbx")) return "fbx";
  if (name.endsWith(".glb") || name.endsWith(".gltf")) return "gltf";
  if (name.endsWith(".obj")) return "obj";
  throw new Error(`Unsupported format: ${name || source}`);
}

function disposeMaterial(material: THREE.Material): void {
  for (const value of Object.values(material)) {
    if (value instanceof THREE.Texture) {
      value.dispose();
    }
  }
  material.dispose();
}

function disposeObject3D(object: THREE.Object3D): void {
  object.traverse((node) => {
    if (!(node instanceof THREE.Mesh)) return;
    node.geometry?.dispose();
    const { material } = node;
    if (Array.isArray(material)) {
      material.forEach(disposeMaterial);
    } else if (material) {
      disposeMaterial(material);
    }
  });
}

const SKIN_COLOR = 0xd4a574;
const DEFAULT_GRAY = new Set([0xffffff, 0xcccccc, 0x808080, 0xaaaaaa, 0x999999]);

function materialHasTexture(material: THREE.Material): boolean {
  const mat = material as THREE.MeshStandardMaterial;
  return !!(mat.map || mat.normalMap || mat.emissiveMap || mat.aoMap);
}

function isUntexturedDefault(material: THREE.Material): boolean {
  if (materialHasTexture(material)) return false;
  const mat = material as THREE.MeshStandardMaterial & { color?: THREE.Color };
  if (!mat.color) return true;
  return DEFAULT_GRAY.has(mat.color.getHex());
}

function createUnlitTextureMaterial(map: THREE.Texture): THREE.MeshBasicMaterial {
  map.colorSpace = THREE.SRGBColorSpace;
  map.minFilter = THREE.LinearFilter;
  map.magFilter = THREE.LinearFilter;
  map.generateMipmaps = false;
  return new THREE.MeshBasicMaterial({
    map,
    toneMapped: false,
    transparent: false,
  });
}

function createSkinMaterial(): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color: SKIN_COLOR,
    roughness: 0.52,
    metalness: 0.03,
  });
}

function modelHasDiffuseMap(object: THREE.Object3D): boolean {
  let found = false;
  object.traverse((node) => {
    if (found || !(node instanceof THREE.Mesh)) return;
    const mats = Array.isArray(node.material) ? node.material : [node.material];
    for (const mat of mats) {
      if (mat && materialHasTexture(mat)) {
        found = true;
        break;
      }
    }
  });
  return found;
}

function applyDiffuseTexture(object: THREE.Object3D, texture: THREE.Texture): void {
  object.traverse((node) => {
    if (!(node instanceof THREE.Mesh)) return;
    const applyOne = (material: THREE.Material): THREE.Material => {
      material.dispose();
      return createUnlitTextureMaterial(texture);
    };
    if (Array.isArray(node.material)) {
      node.material = node.material.map(applyOne);
    } else if (node.material) {
      node.material = applyOne(node.material);
    } else {
      node.material = createUnlitTextureMaterial(texture);
    }
  });
}

function enhanceMeshAppearance(object: THREE.Object3D): void {
  object.traverse((node) => {
    if (!(node instanceof THREE.Mesh)) return;

    const geometry = node.geometry;
    if (geometry) {
      geometry.computeVertexNormals();
    }

    const upgradeMaterial = (material: THREE.Material): THREE.Material => {
      if (materialHasTexture(material)) {
        // 3DGS 烘焙贴图 = albedo，与 splat 一样应无光照显示（勿用 Standard + 灯光）
        if (material instanceof THREE.MeshStandardMaterial && material.map) {
          const map = material.map;
          material.dispose();
          return createUnlitTextureMaterial(map);
        }
        if (material instanceof THREE.MeshPhongMaterial && material.map) {
          const map = material.map;
          material.dispose();
          return createUnlitTextureMaterial(map);
        }
        if (material instanceof THREE.MeshLambertMaterial && material.map) {
          const map = material.map;
          material.dispose();
          return createUnlitTextureMaterial(map);
        }
        return material;
      }

      if (material instanceof THREE.MeshStandardMaterial && !isUntexturedDefault(material)) {
        material.roughness = material.roughness ?? 0.52;
        material.metalness = material.metalness ?? 0.03;
        return material;
      }

      material.dispose();
      return createSkinMaterial();
    };

    if (Array.isArray(node.material)) {
      node.material = node.material.map(upgradeMaterial);
    } else if (node.material) {
      node.material = upgradeMaterial(node.material);
    } else {
      node.material = createSkinMaterial();
    }
  });
}

export class ModelViewer {
  private readonly host: HTMLElement;
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly controls: OrbitControls;
  private readonly clock = new THREE.Clock();
  private readonly options: ModelViewerOptions;

  private root: THREE.Object3D | null = null;
  private mixer: THREE.AnimationMixer | null = null;
  private activeAction: THREE.AnimationAction | null = null;
  private clips: THREE.AnimationClip[] = [];
  private playing = true;
  private raf = 0;
  private resizeObserver: ResizeObserver | null = null;

  constructor(host: HTMLElement, options: ModelViewerOptions = {}) {
    this.host = host;
    this.options = options;

    this.scene.background = new THREE.Color(0x0f1117);

    const width = host.clientWidth || 640;
    const height = host.clientHeight || 480;
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 1000);
    this.camera.position.set(0, 1.2, 2.8);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(width, height);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.shadowMap.enabled = true;
    host.appendChild(this.renderer.domElement);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0.9, 0);
    this.controls.update();

    this.addLights();
    this.addGround();

    this.resizeObserver = new ResizeObserver(() => this.handleResize());
    this.resizeObserver.observe(host);
    window.addEventListener("resize", this.handleResize);

    this.startLoop();
  }

  private addLights(): void {
    const hemi = new THREE.HemisphereLight(0xfff4ea, 0x2a3142, 0.65);
    this.scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 1.25);
    key.position.set(3, 6, 4);
    key.castShadow = true;
    this.scene.add(key);

    const fill = new THREE.DirectionalLight(0x9ecbff, 0.45);
    fill.position.set(-4, 2, -2);
    this.scene.add(fill);

    const rim = new THREE.DirectionalLight(0xffe0c0, 0.35);
    rim.position.set(0, 3, -5);
    this.scene.add(rim);
  }

  private addGround(): void {
    const grid = new THREE.GridHelper(10, 20, 0x3d4f6f, 0x252b38);
    grid.position.y = 0;
    const gridMats = Array.isArray(grid.material) ? grid.material : [grid.material];
    for (const mat of gridMats) {
      mat.opacity = 0.35;
      mat.transparent = true;
    }
    this.scene.add(grid);
  }

  private handleResize = (): void => {
    const width = this.host.clientWidth;
    const height = this.host.clientHeight;
    if (width <= 0 || height <= 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  };

  private startLoop(): void {
    const tick = () => {
      this.raf = requestAnimationFrame(tick);
      const delta = this.clock.getDelta();
      this.controls.update();
      if (this.playing) {
        this.mixer?.update(delta);
      }
      this.renderer.render(this.scene, this.camera);
    };
    tick();
  }

  async loadFromFile(file: File): Promise<LoadedModel> {
    const url = URL.createObjectURL(file);
    try {
      return await this.loadFromUrl(url, file.name);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  async loadFromUrl(url: string, fileName?: string, options?: { externalTextureUrl?: string }): Promise<LoadedModel> {
    const format = inferFormat(url, fileName);
    const displayName = fileName || url.split("/").pop()?.split("?")[0] || "model";

    let root: THREE.Object3D;
    let animations: THREE.AnimationClip[] = [];

    if (format === "fbx") {
      const loader = new FBXLoader();
      root = await loader.loadAsync(url);
      animations = root.animations ?? [];
    } else if (format === "gltf") {
      const loader = new GLTFLoader();
      const gltf = await loader.loadAsync(url);
      root = gltf.scene;
      animations = gltf.animations ?? [];
    } else {
      const loader = new OBJLoader();
      root = await loader.loadAsync(url);
      animations = [];
    }

    if (options?.externalTextureUrl && !modelHasDiffuseMap(root)) {
      const texLoader = new THREE.TextureLoader();
      texLoader.setCrossOrigin("anonymous");
      try {
        const texture = await texLoader.loadAsync(options.externalTextureUrl);
        applyDiffuseTexture(root, texture);
      } catch {
        /* 无贴图时仍走 enhanceMeshAppearance 的肤色回退 */
      }
    }

    this.setModel(root, animations, displayName);
    if (modelHasDiffuseMap(root)) {
      this.renderer.toneMapping = THREE.NoToneMapping;
    } else {
      this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    }
    return { root, animations, displayName };
  }

  private setModel(root: THREE.Object3D, animations: THREE.AnimationClip[], _displayName: string): void {
    this.clearModel();

    this.root = root;
    enhanceMeshAppearance(root);
    root.traverse((node) => {
      if (node instanceof THREE.Mesh) {
        node.castShadow = true;
        node.receiveShadow = true;
      }
    });
    this.scene.add(root);
    this.fitCameraToObject(root);

    this.clips = animations;
    if (animations.length > 0) {
      this.mixer = new THREE.AnimationMixer(root);
      this.playClip(0);
    } else {
      this.mixer = null;
      this.activeAction = null;
    }
    this.options.onAnimationsChange?.(this.clips);
  }

  private clearModel(): void {
    if (this.activeAction) {
      this.activeAction.stop();
      this.activeAction = null;
    }
    this.mixer = null;
    this.clips = [];
    if (this.root) {
      this.scene.remove(this.root);
      disposeObject3D(this.root);
      this.root = null;
    }
    this.options.onAnimationsChange?.([]);
  }

  private fitCameraToObject(object: THREE.Object3D): void {
    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty()) return;

    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 0.001);
    const distance = maxDim * 1.8;

    this.controls.target.copy(center);
    this.camera.position.set(center.x + distance * 0.35, center.y + maxDim * 0.55, center.z + distance);
    this.camera.near = Math.max(maxDim / 200, 0.01);
    this.camera.far = Math.max(maxDim * 50, 100);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  getAnimationClips(): THREE.AnimationClip[] {
    return this.clips;
  }

  playClip(index: number): void {
    if (!this.mixer || index < 0 || index >= this.clips.length) return;

    const next = this.mixer.clipAction(this.clips[index]);
    if (this.activeAction && this.activeAction !== next) {
      this.activeAction.fadeOut(0.25);
    }
    next.reset().fadeIn(0.25).play();
    this.activeAction = next;
    this.setPlaying(true);
  }

  setPlaying(playing: boolean): void {
    this.playing = playing;
    if (this.activeAction) {
      this.activeAction.paused = !playing;
    }
    this.options.onPlayingChange?.(playing);
  }

  togglePlaying(): boolean {
    this.setPlaying(!this.playing);
    return this.playing;
  }

  isPlaying(): boolean {
    return this.playing;
  }

  dispose(): void {
    cancelAnimationFrame(this.raf);
    window.removeEventListener("resize", this.handleResize);
    this.resizeObserver?.disconnect();
    this.clearModel();
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}

export { inferFormat };
