import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  root: ".",
  server: {
    port: 5175,
    strictPort: true,
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
