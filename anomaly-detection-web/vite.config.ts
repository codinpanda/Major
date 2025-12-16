import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    chunkSizeWarningLimit: 1000, // Increase warning limit to 1000kB (default 500kB)
    rollupOptions: {
      output: {
        manualChunks: {
          'onnxruntime': ['onnxruntime-web'], // Separate ONNX runtime into its own chunk
        },
      },
    },
  },
})
