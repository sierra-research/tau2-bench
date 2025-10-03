import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Note: base path configuration may need adjustment based on deployment strategy
  // Previously configured for standalone GitHub Pages deployment
  base: process.env.NODE_ENV === 'production' ? (process.env.GITHUB_PAGES_BASE || '/') : '/',
  build: {
    outDir: 'dist'
  }
})
