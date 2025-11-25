// @ts-check
import { defineConfig } from "astro/config";

import favicons from "astro-favicons";

import playformInline from "@playform/inline";

import tailwindcss from "@tailwindcss/vite";

import icon from "astro-icon";

export default defineConfig({
  vite: {
    plugins: [tailwindcss()],
  },

  integrations: [
    favicons({
      name: "ASL 2 English & English 2 ASL",
      short_name: "ASL Research",
      themes: ["#f9fafb", "#040506"],
    }),
    playformInline(),
    icon(),
  ],
});
