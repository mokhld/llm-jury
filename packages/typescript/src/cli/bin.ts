#!/usr/bin/env node

// The installed `llm-jury` executable. npm links it into node_modules/.bin as a
// symlink, so this file runs the CLI unconditionally rather than comparing its
// own path with process.argv[1].
import { runCli } from "./main.ts";

process.exitCode = await runCli();
