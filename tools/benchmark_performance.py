"""
Command-line benchmark harness for the Cricket Captain fast-processing pipeline.

Run with:
    python -m tools.benchmark_performance
"""

import time
import tempfile
from pathlib import Path

from fast_processing import FastCricketProcessor
from game import process_game_stats
from bat import process_bat_stats
from bowl import process_bowl_stats


def time_standard_processing(temp_dir, logger=None):
    """Time the standard processing approach."""
    log = logger or print
    log("🐌 Testing Standard Processing...")
    start = time.time()

    try:
        process_game_stats(temp_dir)
        log(f"   Standard Game Stats: {time.time() - start:.2f}s")

        bowl_start = time.time()
        process_bowl_stats(temp_dir)
        log(f"   Standard Bowl Stats: {time.time() - bowl_start:.2f}s")

        bat_start = time.time()
        process_bat_stats(temp_dir)
        log(f"   Standard Bat Stats: {time.time() - bat_start:.2f}s")

    except Exception as exc:
        log(f"   Error in standard processing: {exc}")
        return None

    total = time.time() - start
    log(f"   Standard Total: {total:.2f}s")
    return total


def time_fast_processing(temp_dir, logger=None):
    """Time the fast processing approach."""
    log = logger or print
    log("🚀 Testing Fast Processing...")
    start = time.time()

    try:
        processor = FastCricketProcessor()
        processor.process_all_scorecards(temp_dir)

    except Exception as exc:
        log(f"   Error in fast processing: {exc}")
        return None

    total = time.time() - start
    log(f"   Fast Total: {total:.2f}s")
    return total


def run_benchmark(temp_dir=None, logger=None):
    """Run both benchmark passes and return timing statistics."""
    log = logger or print
    log("⚡ Cricket Captain Performance Benchmark")
    log("=" * 50)

    target_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "cricket_scorecards"

    if not target_dir.exists():
        log("❌ No test data found. Run the main processing first.")
        return {
            "standard_time": None,
            "fast_time": None,
            "speedup": None,
            "file_count": 0,
        }

    txt_files = list(target_dir.glob("*.txt"))
    log(f"📂 Testing with {len(txt_files)} scorecard files")
    log("")

    standard_time = time_standard_processing(str(target_dir), logger=log)
    log("")
    fast_time = time_fast_processing(str(target_dir), logger=log)
    log("")

    speedup = None
    if standard_time and fast_time:
        speedup = standard_time / fast_time if fast_time else None
        log("🏆 PERFORMANCE RESULTS:")
        log(f"   Standard Processing: {standard_time:.2f}s")
        log(f"   Fast Processing:     {fast_time:.2f}s")
        log(f"   Speed Improvement:   {speedup:.1f}x faster!")

        if speedup is not None:
            if speedup >= 2.0:
                log("   ✅ EXCELLENT: Achieved 2x+ speed improvement!")
            elif speedup >= 1.5:
                log("   ✅ GOOD: Significant speed improvement achieved!")
            else:
                log("   ⚠️  MODEST: Some improvement, but could be better")
    else:
        log("❌ Could not complete benchmark comparison")

    return {
        "standard_time": standard_time,
        "fast_time": fast_time,
        "speedup": speedup,
        "file_count": len(txt_files),
    }


def run_release_benchmark(temp_dir=None):
    """Run the benchmark and capture logs for UI presentation."""
    captured_logs = []

    def _capture(message: str):
        captured_logs.append(message)

    results = run_benchmark(temp_dir=temp_dir, logger=_capture)
    results["logs"] = captured_logs
    return results


def main():
    run_benchmark()


if __name__ == "__main__":
    main()
