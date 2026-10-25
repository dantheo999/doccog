import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

declare const process: { env: Record<string, string | undefined> };

export default defineConfig({
  plugins: [react()],
  // 8010 avoids the host port 8000 that a local Unstract platform publishes.
  server: { proxy: { "/api": process.env.DOCCOG_API ?? "http://127.0.0.1:8010" } },
  test: { environment: "jsdom", setupFiles: ["./src/test-setup.ts"], restoreMocks: true },
});
