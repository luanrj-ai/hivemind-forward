"""Full MVP run: 15 days × 5 personas × 3 modes via claude CLI."""

import sys
import time
from simulation import run_simulation, save_result


def main():
    start = time.time()
    total_steps = [0]

    def on_step(idx, total, msg):
        total_steps[0] = total
        elapsed = time.time() - start
        eta = (elapsed / idx) * (total - idx) if idx > 0 else 0
        print(f"[{idx:3d}/{total}] {msg} | elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    print("=== Full MVP run starting ===", flush=True)
    print("Window: 2026-04-14 → 2026-05-02 (~15 trading days)", flush=True)
    print("Personas: 5  |  Modes: α β γ", flush=True)
    print("Backend: claude_cli  |  Parallelism: 5", flush=True)
    print("Expected: ~225 LLM calls, ~$7, ~10-15 min", flush=True)
    print("", flush=True)

    result = run_simulation(
        ticker="AAPL",
        start_date="2026-04-14",
        end_date="2026-05-02",
        sensitivity=0.3,
        modes=("alpha", "beta", "gamma"),
        use_llm_cache=True,
        mock=False,
        backend="claude_cli",
        max_concurrency=5,
        on_step=on_step,
    )

    elapsed = time.time() - start
    save_result(result, "latest")
    save_result(result, "full_run")

    print("", flush=True)
    print(f"=== Done in {elapsed/60:.1f} min ===", flush=True)
    print("", flush=True)
    print("Final PnL summary per mode:", flush=True)
    for m in result.modes:
        pnls = []
        for pid, pdat in result.final_portfolios[m].items():
            pnls.append((pid, pdat["pnl_pct"]))
        pnls.sort(key=lambda x: -x[1])
        print(f"  Mode {m}:", flush=True)
        for pid, p in pnls:
            print(f"    {pid:32s} {p:+6.2f}%", flush=True)
        print("", flush=True)

    print("Real vs virtual final price by mode:", flush=True)
    last_day_records = [d for d in result.days if d.date == result.days[-1].date]
    for d in last_day_records:
        delta = (d.virtual_close - d.real_close) / d.real_close * 100
        print(f"  Mode {d.mode}: real={d.real_close:.2f} virtual={d.virtual_close:.2f} delta={delta:+.2f}%", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
