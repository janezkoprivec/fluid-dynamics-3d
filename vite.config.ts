import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [
    {
      // Force full page reload on any change. HMR re-evaluates modules without
      // tearing down the WebGPU device/canvas context, which crashes Firefox
      // Nightly. A full reload runs `beforeunload` cleanup and starts fresh.
      name: 'webgpu-full-reload',
      handleHotUpdate({ server }) {
        server.ws.send({ type: 'full-reload' });
        return [];
      },
    },
  ],
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
  build: {
    target: 'es2022',
    sourcemap: true,
  },
});
