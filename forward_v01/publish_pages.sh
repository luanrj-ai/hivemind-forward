#!/bin/zsh
# Refresh the public GitHub Pages site (gh-pages branch) from the current static
# files. Uses a dedicated git worktree (a sibling dir under $HOME, never under
# ~/Desktop) so it never disturbs the main checkout and dodges macOS TCC.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
REPO="${0:A:h:h}"
SRC="$REPO/forward_v01"
PAGES="$REPO/../$(basename "$REPO")-pages"   # e.g. ~/forward-sim-pages

cd "$REPO" || exit 0
git fetch -q origin gh-pages 2>/dev/null || true
if [ ! -e "$PAGES/.git" ]; then
  git worktree add -f -B gh-pages "$PAGES" origin/gh-pages 2>/dev/null \
    || git worktree add -f "$PAGES" gh-pages 2>/dev/null \
    || { echo "WARN: could not set up gh-pages worktree"; exit 0; }
fi
cd "$PAGES" || exit 0
git pull -q --ff-only origin gh-pages 2>/dev/null || true

mkdir -p results
cp "$SRC/landing.html" index.html      # Pages root = landing
cp "$SRC/landing.html" landing.html    # explore's "← 首页" target
cp "$SRC/explore.html" explore.html
cp "$SRC/onepager.html" onepager.html
cp "$SRC/results/viz_data.js" results/viz_data.js
cp "$SRC/results/viz_data.json" results/viz_data.json
touch .nojekyll

git add -A
if git diff --cached --quiet; then
  echo "pages: nothing changed"
else
  git -c user.name="Julie Luan" -c user.email="chenxia2@uoem.edu.gr" \
    commit -q -m "pages: refresh $(date +%F)"
  git push -q origin gh-pages && echo "✓ live site refreshed: https://luanrj-ai.github.io/hivemind-forward/" \
    || echo "WARN: pages push failed (committed locally)"
fi
