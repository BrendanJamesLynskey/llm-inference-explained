import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@content": path.resolve(__dirname, "./content"),
    },
  },
  test: {
    globals: true,
    environment: "node",
    include: ["tests/unit/**/*.test.ts"],
    coverage: {
      provider: "v8",
      reporter: ["text", "html", "lcov"],
      include: ["src/lib/**/*.ts"],
      exclude: [
        "src/lib/**/*.test.ts",
        // Type-only modules: no runtime exports → nothing for v8 to cover.
        "src/lib/transformer/types.ts",
        // MDX wiring: components map and FS loader. Exercised by the
        // chapter e2e tests in tests/e2e/chapters.spec.ts.
        "src/lib/mdx/sections.ts",
        "src/lib/mdx/components.ts",
      ],
      thresholds: {
        // Vendored from transformer-explainer, which mandates 100% line
        // coverage for src/lib/transformer/ (its CLAUDE.md §8). The KV cache
        // added here is held to the same bar.
        "src/lib/transformer/**": {
          lines: 100,
          functions: 100,
          statements: 100,
          branches: 80,
        },
        // The calculators behind the interactives.
        "src/lib/inference/**": {
          lines: 95,
          functions: 95,
          statements: 95,
          branches: 85,
        },
        // The engine wrapper, hand-off closed forms and simulator presets
        // (the vendored engine itself is .js, tested by parity, not counted).
        "src/lib/disagg/**": {
          lines: 95,
          functions: 95,
          statements: 95,
          branches: 85,
        },
        lines: 80,
        functions: 80,
        branches: 80,
        statements: 80,
      },
    },
  },
});
