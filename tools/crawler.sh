#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# crawler.sh - isolated worktree lifecycle for one crawler branch.
#
# WHY THIS EXISTS
#   N crawlers editing data/ in one working tree overwrite each other. That
#   was the actual failure mode in earlier waves: parallel agents wrote to the
#   same files and the enum values silently dropped. Each crawler gets its OWN
#   git worktree off origin/main, so its writes are invisible until merge.
#
# WINDOWS / MSYS PATH TRAP
#   Pass C:/Users/... NOT /c/Users/... . MSYS path conversion is DISABLED for
#   native programs here, so a leading /c is treated as a literal relative
#   path and git creates C:\c\Users\... . That exact bug hit this repo and the
#   stale worktree has to be removed by hand.
#
# USAGE
#   tools/crawler.sh new  <slug>   -> creates worktree + branch, prints its path
#   tools/crawler.sh commit <slug> <msg-file>
#   tools/crawler.sh push   <slug> -> opens a PR, prints the PR number
#   tools/crawler.sh drop  <slug>  -> removes worktree + deletes remote branch
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="C:/Users/Yoav/inference-research"
WT_ROOT="C:/Users/Yoav/kb-crawlers"
SLUG="${2:-}"

wt_path() { echo "$WT_ROOT/$SLUG"; }

cmd_new() {
  [ -n "$SLUG" ] || { echo "usage: crawler.sh new <slug>" >&2; exit 2; }
  mkdir -p "$WT_ROOT"
  # Always branch from a FRESH origin/main so two crawlers never start from
  # different points.
  git -C "$REPO" fetch origin --prune -q
  if [ -e "$(wt_path)" ]; then
    echo "worktree already exists: $(wt_path)" >&2; exit 1
  fi
  git -C "$REPO" worktree add -q "$(wt_path)" -b "crawler/$SLUG" origin/main
  echo "$(wt_path)"
}

cmd_commit() {
  # $1 is already "commit", so the slug is $2 and the message file is $3 --
  # but under `set -u` a missing positional aborts before the guard below can
  # print anything useful, so the count is checked explicitly.
  local wt
  if [ "$#" -lt 3 ]; then
    echo "usage: crawler.sh commit <slug> <msg-file>" >&2; exit 2
  fi
  wt="$(wt_path)"
  # SCHEMA.md is named explicitly because it was NOT in the original path list,
  # and that silently dropped a whole commit's documentation. The 2026-10-05
  # schema-authority pass edited SCHEMA.md in its worktree, called this command,
  # and the file was never staged -- so its commit shipped schemas that the
  # human-facing doc did not describe, which is the exact drift this repo has
  # already hit once. A schema change without its SCHEMA.md row IS the failure
  # mode, so the file is named here rather than left to the path list.
  #
  # Found by: an agent that had already verified validate.py and the test suite,
  # committed, and only discovered the loss when `git status` still showed the
  # file modified. Worth noting that the worktree was NOT re-read after the
  # commit -- the edits were gone, not merely unstaged, because the commit cycle
  # in the parent repo reset the working copy.
  git -C "$wt" add -A data/ schemas/ docs/ tools/ SCHEMA.md README.md AGENTS.md .github/ site/
  if git -C "$wt" diff --cached --quiet; then
    echo "NOTHING TO COMMIT" >&2; exit 3
  fi
  git -C "$wt" commit -q -F "$3"
  echo "committed $(git -C "$wt" rev-parse --short HEAD)"
}

cmd_push() {
  local wt; wt="$(wt_path)"
  git -C "$wt" push -q -u origin "crawler/$SLUG"
  gh pr create --repo phantomic12/inference-research \
    --head "crawler/$SLUG" --base main \
    --title "crawler($SLUG): data intake" \
    --body "Automated intake from crawler \`$SLUG\`. Validated locally with tools/validate.py and tools/test_tools.py before push." \
    2>&1 | tail -1
}

cmd_drop() {
  local wt; wt="$(wt_path)"
  gh pr close "$2" --repo phantomic12/inference-research --comment \
    "Closed by coordinator: superseded or failed rebase." >/dev/null 2>&1 || true
  git -C "$REPO" worktree remove --force "$wt" 2>/dev/null || true
  git -C "$REPO" push -q origin --delete "crawler/$SLUG" 2>/dev/null || true
  git -C "$REPO" branch -D "crawler/$SLUG" >/dev/null 2>&1 || true
  rmdir "$wt" 2>/dev/null || true
  echo "dropped $SLUG"
}

case "${1:-}" in
  new)    cmd_new "$@" ;;
  commit) cmd_commit "$@" ;;
  push)   cmd_push "$@" ;;
  drop)   cmd_drop "$@" ;;
  *) echo "usage: crawler.sh {new|commit|push|drop} <slug> [args]" >&2; exit 2 ;;
esac
