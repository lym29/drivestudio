#!/bin/bash
set -euo pipefail

export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [[ -s "$NVM_DIR/nvm.sh" ]]; then
  # shellcheck source=/dev/null
  source "$NVM_DIR/nvm.sh"
  nvm use 20 >/dev/null
fi

SUPERSPLAT_DIR="$(cd "$(dirname "$0")/../supersplat" && pwd)"
PORT="${PORT:-3000}"

if [[ ! -d "$SUPERSPLAT_DIR/node_modules" ]]; then
  echo "SuperSplat dependencies missing. Run: cd $SUPERSPLAT_DIR && npm install"
  exit 1
fi

cd "$SUPERSPLAT_DIR"
echo "Starting SuperSplat at http://localhost:${PORT}"
echo "Open the URL, then drag & drop a .ply file (e.g. scene_000/models/background.ply)"
exec npm run develop
