// @ts-check
import { defineConfig } from "astro/config";

export default defineConfig({
  vite: {
    css: {
      transformer: "lightningcss",
      lightningcss: {
        drafts: { customMedia: true },
      },
    },
    build: {
      cssMinify: "lightningcss",
    },
  },
});
