#!/usr/bin/env bash
# fetch_demo_videos.sh — download OpenReality demo videos into server/demo_videos/
#
# The demo videos (e.g. office_loop.mp4, ~349 MB total) were removed from git
# to keep the repo lean.  This script fetches them from cloud storage and places
# them in the expected directory so that:
#
#   modal run modal_data_export_test.py --demo-video office_loop.mp4 --to-lerobot
#   modal run modal_streaming.py::upload_demo_videos
#
# ... both find the files.
#
# ── REQUIRED: set DEMO_VIDEOS_BASE_URL ─────────────────────────────────────────
#
#   export DEMO_VIDEOS_BASE_URL="https://your-bucket.s3.amazonaws.com/demo_videos"
#
# The URL must point to the directory (no trailing slash) that contains the
# individual .mp4 files.  Each file is fetched as:
#
#   $DEMO_VIDEOS_BASE_URL/<filename>.mp4
#
# Supported storage backends — pick whichever you have credentials for:
#
#   • AWS S3 / S3-compatible (presigned URL or public):
#       export DEMO_VIDEOS_BASE_URL="https://my-bucket.s3.us-east-1.amazonaws.com/demo_videos"
#
#   • Google Cloud Storage (public or with HMAC via presigned URL):
#       export DEMO_VIDEOS_BASE_URL="https://storage.googleapis.com/my-bucket/demo_videos"
#
#   • Any HTTPS URL (Cloudflare R2, Backblaze B2 public bucket, …):
#       export DEMO_VIDEOS_BASE_URL="https://pub-xxx.r2.dev/demo_videos"
#
#   For authenticated endpoints generate per-file presigned URLs and override the
#   DEMO_VIDEOS_* variables below, or use rclone/aws-cli directly.
#
# ── DEMO VIDEO LIST ─────────────────────────────────────────────────────────────
# Add or remove filenames as the set of demo videos changes.
DEMO_VIDEOS=(
    "office_loop.mp4"
)
# ────────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Resolve destination directory ───────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST_DIR="$REPO_ROOT/server/demo_videos"

# ── Validate required env var ───────────────────────────────────────────────────
if [ -z "${DEMO_VIDEOS_BASE_URL:-}" ]; then
    echo ""
    echo "ERROR: DEMO_VIDEOS_BASE_URL is not set."
    echo ""
    echo "  Set it to the base URL of your cloud storage directory, e.g.:"
    echo "    export DEMO_VIDEOS_BASE_URL='https://your-bucket.s3.amazonaws.com/demo_videos'"
    echo "  Then re-run this script."
    echo ""
    echo "  See scripts/fetch_demo_videos.sh for full documentation."
    exit 1
fi

# Strip trailing slash so URL construction is consistent
BASE_URL="${DEMO_VIDEOS_BASE_URL%/}"

# ── Create destination directory ─────────────────────────────────────────────────
mkdir -p "$DEST_DIR"
echo "Destination: $DEST_DIR"
echo "Source base: $BASE_URL"
echo ""

# ── Download ──────────────────────────────────────────────────────────────────────
DOWNLOADED=0
SKIPPED=0
FAILED=0

for video in "${DEMO_VIDEOS[@]}"; do
    dest_file="$DEST_DIR/$video"
    src_url="$BASE_URL/$video"

    if [ -f "$dest_file" ]; then
        echo "[skip]   $video (already exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "[fetch]  $video"
    if curl --fail --silent --show-error --location \
            --output "$dest_file" \
            "$src_url"; then
        size=$(du -sh "$dest_file" | cut -f1)
        echo "[ok]     $video ($size)"
        DOWNLOADED=$((DOWNLOADED + 1))
    else
        echo "[error]  $video — download failed from $src_url"
        rm -f "$dest_file"   # remove partial download
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "Done: $DOWNLOADED downloaded, $SKIPPED skipped, $FAILED failed."

if [ "$FAILED" -gt 0 ]; then
    echo "Some downloads failed.  Check that DEMO_VIDEOS_BASE_URL is correct and accessible."
    exit 1
fi
