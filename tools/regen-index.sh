#!/usr/bin/env bash
# Regenerate docs/00-index.md from the COMMITTED tree and commit the result.
#
# Why this exists: with many crawler agents writing to data/ concurrently, the
# working tree holds records that are not in any commit. Running tools/index.py
# in the working tree produces an index describing a record set that does not
# exist on the branch, and CI's "Check index is regenerated" step fails on a
# bare off-by-N with no obvious cause. This script runs the indexer inside a
# throwaway worktree checked out at origin/main, so it can only ever see
# committed records.
#
# Usage:  bash tools/regen-index.sh
# Then:   git push origin <branch>
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WT="${TMPDIR:-$LOCALAPPDATA/Temp}/ir-index-$$"

cd "$REPO"
git fetch -q origin

# Refuse to run if the index would be generated against uncommitted work that
# the caller probably forgot about. Warn loudly rather than silently commit a
# mismatched index.
if ! git diff --quiet -- docs/00-index.md; then
  echo "WARNING: docs/00-index.md has uncommitted local edits; they will be overwritten." >&2
fi

cleanup() { git worktree remove --force "$WT" 2>/dev/null || true; git worktree prune; }
trap cleanup EXIT

git worktree add -q --detach "$WT" origin/main
( cd "$WT" && python tools/index.py )

cp "$WT/docs/00-index.md" "$REPO/docs/00-index.md"

cd "$REPO"
git add docs/00-index.md
if git diff --cached --quiet; then
  echo "index already up to date"
else
  git commit -q -m "Regenerate index from the committed tree"
  echo "committed regenerated index"
fi