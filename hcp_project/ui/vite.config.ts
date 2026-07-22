import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/scenarios': 'http://localhost:8000',
      '/scenario': 'http://localhost:8000',
      '/stream': 'http://localhost:8000',
      '/run_hcp': 'http://localhost:8000',
      '/motion_states': 'http://localhost:8000',
      '/metrics': 'http://localhost:8000',
      '/map': 'http://localhost:8000',
      '/audio': 'http://localhost:8000',
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
