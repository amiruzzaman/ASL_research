// @ts-check
import { defineConfig } from "astro/config";

import favicons from "astro-favicons";

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

  integrations: [
    favicons({
      name: "ASL 2 English & English 2 ASL",
      short_name: "ASL Research",
      themes: ["#f9fafb", "#040506"],
    }),
  ],
});
