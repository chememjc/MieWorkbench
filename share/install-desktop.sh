#!/bin/sh
# Install the MieWorkbench desktop entry + icon for the current user.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
APPS="$HOME/.local/share/applications"
mkdir -p "$APPS"
sed "s|@REPO@|$REPO|g" "$REPO/share/mieworkbench.desktop" \
    > "$APPS/mieworkbench.desktop"
chmod +x "$REPO/bin/mieworkbench"
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPS" || true
fi
echo "Installed $APPS/mieworkbench.desktop (Exec -> $REPO/bin/mieworkbench)"
