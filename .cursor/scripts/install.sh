#!/usr/bin/env bash
set -euo pipefail

# Idempotent bootstrap for PlantVillage (greenfield repo).
# Prepares a verified Node toolchain; add package installs here when the app lands.

if ! command -v node >/dev/null 2>&1; then
  echo "error: node is not on PATH" >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "error: npm is not on PATH" >&2
  exit 1
fi

node --version
npm --version

# Smoke check that the runtime can execute project-local logic when added later.
node -e "console.log('PlantVillage install: OK')"
