import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In dev, Vite proxies /api to the backend so the browser only ever talks to one origin.
// Local edition: API on :8020, UI on :5174 (the cloud edition uses 8010/5173, so both can run side by side).
const API_PORT = process.env.KMRL_API_PORT ?? '8020'

export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.KMRL_UI_PORT ?? 5174),
    proxy: { '/api': `http://127.0.0.1:${API_PORT}` },
  },
})
