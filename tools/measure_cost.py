"""Development-only: measure predictor/scorer wall time, CPU time and peak RAM.

psutil is a development-time measurement dependency only; it is NOT required by
the exported agent (requirements.txt still contains just sqlglot). All SQL work
runs in short-lived subprocesses spawned by the official executor, so this
helper aggregates CPU time and resident memory over the whole process tree.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    sys.exit("measure_cost.py needs psutil (development-only, not an export dependency)")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def measure(command, label):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "submission")
    process = subprocess.Popen(command, cwd=str(ROOT), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True)
    root = psutil.Process(process.pid)
    started = time.perf_counter()
    seen_cpu = {}
    peak_tree_mib = 0.0
    peak_single_mib = 0.0
    while process.poll() is None:
        try:
            live = [root] + root.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # A short-lived executor child disappeared while psutil walked the tree.
            live = [root]
        current_mib = 0.0
        for item in live:
            try:
                with item.oneshot():
                    times = item.cpu_times()
                    rss = item.memory_info().rss / 1048576
                cpu = times.user + times.system
                seen_cpu[item.pid] = max(seen_cpu.get(item.pid, 0.0), cpu)
                current_mib += rss
                peak_single_mib = max(peak_single_mib, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        peak_tree_mib = max(peak_tree_mib, current_mib)
        time.sleep(0.02)
    wall = time.perf_counter() - started
    output = process.stdout.read().strip()
    result = {
        "wall_seconds": round(wall, 2),
        "cpu_seconds": round(sum(seen_cpu.values()), 2),
        "peak_tree_rss_mib": round(peak_tree_mib, 1),
        "peak_single_process_rss_mib": round(peak_single_mib, 1),
        "tracked_processes": len(seen_cpu),
        "command": " ".join(str(part) for part in command),
        "stdout": output,
    }
    print(label, json.dumps(result, indent=2))
    return result


def main():
    results = {}
    results["predict_test"] = measure(
        [str(PY), "-m", "mp3_sql.predict", "--split", "test",
         "--method", "submission/submission.py",
         "--output", "outputs/measure-test.json"], "predict_test")
    results["score_test"] = measure(
        [str(PY), "-m", "mp3_sql.score", "--split", "test",
         "--predictions", "outputs/measure-test.json",
         "--output", "outputs/measure-test-score.json"], "score_test")
    results["total_wall_seconds"] = round(
        results["predict_test"]["wall_seconds"] +
        results["score_test"]["wall_seconds"], 2)
    results["total_cpu_seconds"] = round(
        results["predict_test"]["cpu_seconds"] +
        results["score_test"]["cpu_seconds"], 2)
    (OUT / "runtime.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in ("measure-test.json", "measure-test-score.json"):
        (OUT / name).unlink(missing_ok=True)
    print("wrote", OUT / "runtime.json")


if __name__ == "__main__":
    sys.exit(main())
