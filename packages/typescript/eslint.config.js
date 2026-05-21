import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: ["dist/**", "node_modules/**"],
  },
  ...tseslint.configs.recommended,
  {
    rules: {
      // Tests + adapter shims legitimately need `any` for fake clients,
      // dynamic-import shapes, etc. Warn instead of error.
      "@typescript-eslint/no-explicit-any": "warn",
      // The codebase uses `_` prefix as the unused-arg convention.
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],
    },
  },
);
