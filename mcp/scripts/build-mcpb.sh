#!/usr/bin/env bash
# Build the Claude desktop one-click extension (an .mcpb bundle) from this package.
#
#   npm run build:mcpb            # writes dist-mcpb/openreality-mcp-<version>.mcpb
#   npm run build:mcpb -- out/    # custom output directory
#
# The bundle is a zip of: manifest.json, dist/, production node_modules/,
# icon.png, README.md, LICENSE. Users install it by double-clicking the file
# in Claude desktop. Requires Node 20+ and network access (npm ci).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-dist-mcpb}"
VERSION="$(node -p "require('./package.json').version")"
MANIFEST_VERSION="$(node -p "require('./manifest.json').version")"
if [ "$VERSION" != "$MANIFEST_VERSION" ]; then
  echo "package.json version ($VERSION) != manifest.json version ($MANIFEST_VERSION)" >&2
  exit 1
fi

npm run build

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$OUT"

cp -R dist package.json package-lock.json manifest.json README.md LICENSE vendor "$STAGE"/
[ -f icon.png ] && cp icon.png "$STAGE"/
(cd "$STAGE" && npm ci --omit=dev --ignore-scripts --no-audit --no-fund >/dev/null)

npx -y @anthropic-ai/mcpb validate "$STAGE/manifest.json"
npx -y @anthropic-ai/mcpb pack "$STAGE" "$OUT/openreality-mcp-$VERSION.mcpb"
echo "wrote $OUT/openreality-mcp-$VERSION.mcpb"
