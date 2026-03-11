#!/bin/bash
# Sync content from learn-claude-code to paper-notes/content/
SOURCE="/Users/donglinkang/Documents/projects/learn-claude-code"
DEST="$(cd "$(dirname "$0")" && pwd)/content"

rm -rf "$DEST/knowledge" "$DEST/areas" "$DEST/insights"
cp -r "$SOURCE/knowledge" "$DEST/"
cp -r "$SOURCE/areas" "$DEST/"
cp -r "$SOURCE/insights" "$DEST/"

echo "Content synced from $SOURCE"
