#!/bin/zsh
# forward_v01 daily driver for launchd (local, claude subscription).
# Scores matured predictions, predicts today (300 agents × 5 tickers, paced
# across claude quota windows — can run for hours), updates the scoreboard, and
# commits+pushes results on the forward-predict branch.
#
# launchd gives a minimal PATH, so set it explicitly (claude lives in ~/.local/bin).
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export ANALYST_LLM=claude

# Repo root derived from this script's own location ($0), so the same script
# works whether it lives in the Desktop checkout or the scheduled ~/forward-sim
# clone. (zsh: :A = absolute, :h = dirname.)
REPO="${0:A:h:h}"
cd "$REPO" || exit 1

mkdir -p forward_v01/results/logs
LOG="forward_v01/results/logs/run_$(date +%Y%m%d_%H%M).log"

{
  echo "=== $(date) — forward_v01 daily START (repo: $REPO) ==="
  # Sync latest committed code/results before running (no-op on a clean clone).
  git pull --ff-only origin forward-predict 2>&1 || echo "WARN: git pull skipped/failed"
  # caffeinate -i keeps the Mac awake for the duration of the (long) run.
  caffeinate -i "$REPO/.venv/bin/python" -m forward_v01.daily --agents 300
  echo "--- scoreboard ---"
  "$REPO/.venv/bin/python" -m forward_v01.scoreboard --json

  echo "--- commit & push results ---"
  git add forward_v01/results/pending.jsonl forward_v01/results/scored.jsonl \
          forward_v01/results/scoreboard.json 2>/dev/null
  if git diff --cached --quiet; then
    echo "nothing to commit"
  else
    git commit -m "forward_v01 daily results $(date +%Y-%m-%d)" \
      -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
    git push origin forward-predict || echo "WARN: push failed (results committed locally)"
  fi
  echo "=== $(date) — forward_v01 daily DONE ==="
} >> "$LOG" 2>&1
