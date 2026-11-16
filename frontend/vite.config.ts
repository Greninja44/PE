import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the FastAPI backend so the browser only talks to one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: { "/api": { target: process.env.PE_API_URL ?? "http://127.0.0.1:8000", changeOrigin: true } },
  },
  preview: {
    port: 4173,
    proxy: { "/api": { target: process.env.PE_API_URL ?? "http://127.0.0.1:8000", changeOrigin: true } },
  },
  build: { chunkSizeWarningLimit: 1200 },
});
