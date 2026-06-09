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
  # caffeinate -i keeps the Mac awake for the duration of the run.
  # 80 agents × 5 tickers = 400 calls. The claude subscription's ~5h rolling
  # window empirically allows only ~470 calls (3 runs on 06-09 all capped there
  # and dropped the last 2 tickers), so 750 doesn't fit but 400 completes all 5
  # in ONE window — a complete, once-and-frozen daily prediction beats 150-on-3.
  caffeinate -i "$REPO/.venv/bin/python" -m forward_v01.daily --agents 80 --model haiku
  echo "--- scoreboard ---"
  "$REPO/.venv/bin/python" -m forward_v01.scoreboard --json
  echo "--- rebuild viz data ---"
  "$REPO/.venv/bin/python" -m forward_v01.export_viz --date-stamp "$(date +%F)" || echo "WARN: export_viz failed"

  echo "--- commit & push results ---"
  # Add the whole results dir (logs are *.log → gitignored). Adding specific
  # files breaks when some don't exist yet (scored/scoreboard before any score).
  git add forward_v01/results/
  if git diff --cached --quiet; then
    echo "nothing to commit"
  else
    git commit -m "forward_v01 daily results $(date +%Y-%m-%d)" \
      -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
    git push origin forward-predict || echo "WARN: push failed (results committed locally)"
  fi

  echo "--- refresh public site (gh-pages) ---"
  /bin/zsh "$REPO/forward_v01/publish_pages.sh" || echo "WARN: publish_pages failed"
  echo "=== $(date) — forward_v01 daily DONE ==="
} >> "$LOG" 2>&1
