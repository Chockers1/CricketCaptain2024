# Performance Implementation Guide

Use this runbook to apply the performance improvements that now ship with the Cricket Captain analytics stack. Follow the steps in order, ticking off the validation checklist at the end of each section.

## 1. Memory Optimisation
1. Remove defensive `.copy()` calls when reading from `st.session_state`. Only materialise a copy before mutating a DataFrame.
2. Replace ad-hoc conversions with the helpers provided by `performance_utils.PerformanceManager`:
   - `register_dataframe` for tracking origin and lifetime.
   - `optimize_dataframe_types` to downcast numerical columns and promote low-cardinality strings to `category`.
3. Surface the sidebar monitor:
   ```python
   from performance_utils import perf_manager

   with st.sidebar:
       perf_manager.render_summary()
   ```
4. Add the cleanup buttons to long-lived views so power users can force a cache reset without restarting Streamlit.

**Validation**
- Sidebar shows current memory usage and the five largest cached frames.
- Pressing "Clean session" drops cached frames and re-runs the page without errors.

## 2. Fast Processing Pipeline
1. Import `FastCricketProcessor` and replace the legacy pipeline in ingestion entry points (`views/Home.py`, CLI scripts).
   ```python
   from fast_processing import FastCricketProcessor

   processor = FastCricketProcessor()
   bat_df, bowl_df, metadata = processor.process_all_scorecards(upload_dir)
   ```
2. Keep legacy `process_bat_stats` and `process_bowl_stats` available as fallback for debugging.
3. Cache expensive regex patterns and shared sanitisation utilities inside `fast_processing.py` so both fast and legacy paths stay in sync.
4. Wrap the public fast helpers with `memory_efficient_cache` (`data_types=['match']`) to guarantee fast reruns inside a session.

**Validation**
- `tools/benchmark_performance.py` reports a 2x+ speed-up versus the legacy path on the 3,208-scorecard dataset.
- Uploading the same archive twice rehydrates instantly from cache.

## 3. Polars Adoption
1. When transforming large tables, call `performance_utils.to_polars_frame` to prefer Polars when it is installed, with pandas as a fallback.
2. Use Polars for heavy aggregations (Bat View ranking, Records streaks) and convert the final result back to pandas only when a Streamlit component requires it.
3. Keep vectorised pandas fallbacks for contributors who have not installed Polars locally.

**Validation**
- Aggregation-heavy tabs complete in under a second on the benchmark dataset.
- Unit tests compare pandas vs Polars results for determinism.

## 4. Caching Discipline
1. Tag every cached function with the data types it depends on (`data_types=['batting']`, `['bowling']`, etc.).
2. Invalidate caches via `perf_manager.invalidate(data_types={...})` whenever a new upload replaces source DataFrames.
3. Log cache hits and misses with the provided `memory_efficient_cache` decorators to feed future monitoring.

**Validation**
- Clearing session state invalidates caches and forces recomputation exactly once.
- The cache log shows hit/miss ratios for each data type.

## 5. Quality Gates
1. Add automated smoke tests under `tests/`.
2. Run `python -m tools.benchmark_performance` after major changes and document results in `docs/benchmarks/`.
3. Configure linting (ruff) and type checking (mypy) in pre-commit hooks.

**Validation**
- `pytest` exits cleanly.
- Benchmark timings stay within the agreed tolerance window.

## Appendix: Quick Reference
- **Fast pipeline entry point:** `fast_processing.FastCricketProcessor.process_all_scorecards`.
- **Memory helper:** `performance_utils.PerformanceManager`.
- **Diagnostics:** `tools/fast_processing_diagnostics.py` for regex and processor smoke tests.
- **Documentation:** `docs/performance_optimization_plan.md` (roadmap), `docs/optimization_status_report.md` (change log).
