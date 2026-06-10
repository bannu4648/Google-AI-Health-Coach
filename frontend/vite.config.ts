import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    // Allows ngrok/LAN hostnames when checking the dashboard on a phone.
    allowedHosts: true,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
