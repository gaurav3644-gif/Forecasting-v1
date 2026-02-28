import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const BACKEND = 'http://localhost:8000'

const proxy = (paths: string[]) =>
  Object.fromEntries(paths.map((p) => [p, { target: BACKEND, changeOrigin: true }]))

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    port: 5173,
    proxy: {
      ...proxy([
        '/api',
        '/signin', '/signout', '/signup', '/signin/verify',
        '/auth',
        '/upload',
        '/forecast', '/forecast_status', '/forecast_stop', '/generate_forecast',
        '/supply_plan', '/supply_plan_defaults', '/download_supply_plan',
        '/planning', '/risks', '/actions', '/ai',
        '/history', '/data', '/session', '/admin',
        '/chat', '/connect', '/connector-schema',
        '/request_demo', '/status', '/loading',
        '/static',
      ]),
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
