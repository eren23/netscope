// Flat ESLint config for the extension's TypeScript. typescript-eslint's
// `recommended` set enforces no-explicit-any (locks in the weak-type cleanup) and
// a sane baseline; we only tweak unused-vars to allow an intentional _-prefix.
import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["out/**", "media/**", "node_modules/**", "test/**", "*.mjs"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["src/**/*.ts"],
    rules: {
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
    },
  },
  {
    // Test fixtures build partial graph shapes by hand; `any` is fine there.
    files: ["src/test/**/*.ts"],
    rules: { "@typescript-eslint/no-explicit-any": "off" },
  }
);
