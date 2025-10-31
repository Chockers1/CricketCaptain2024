# Optimization Status Report

_Last updated: 31 Oct 2025_

## Overview
Priority 1 (DataFrame memory management) and the October maintenance sweep are complete. The fast ingestion path now underpins every workflow, and the application comfortably handles archives exceeding 3,000 scorecards while staying within the memory budget.

## Maintenance Highlights
- Extended fast-mode logging to `views/Playerrankings.py` and `views/teamview.py`, including timing checkpoints for filter-heavy routines.
- Silenced pandas 3.x deprecation warnings (`CategoricalDtype`, `Styler.map`, `freq='ME'`, explicit `observed=False`).
- Hardened Records View data preparation to eliminate `SettingWithCopy` warnings triggered by mixed date formats.
- Normalised batting data sanitisation so fast and legacy processors treat suffix noise, categoricals, and partial years consistently.
- Instrumented Bat View with the same stage-level telemetry as Bowling (preprocessing, metrics, render) to aid troubleshooting.
- Benchmarked the fast pipeline against 3,208 scorecards (88,825 batting rows / 40,797 bowling rows) with sub-second extraction and sub-1 s aggregation.

## Technical Changes
| Area | Change | Result |
|------|--------|--------|
| DataFrame handling | Removed redundant `.copy()` calls across core views | 60–70% memory reduction
| Memory monitoring | Added `PerformanceManager` sidebar widget | Real-time usage + one-click cleanup
| Data types | Automated downcasting and categorical promotion | Lower baseline footprint
| Ingestion | Fast path enabled by default with cached regex compilation | 2–5× faster rehydration
| Benchmarks | Added CLI harness and smoke checks | Repeatable regression tracking

## User Impact
- Faster navigation between tabs thanks to cached Polars DataFrames.
- Stable Elo rankings: matrices remain populated while filters highlight the selected subset.
- Memory footprint stays within 200 MB on the benchmark dataset, enabling larger archives without instability.

## Next Steps
See `docs/performance_optimization_plan.md` for the current roadmap (cache lifecycle, Polars migration, automated quality gates, UX polish).

## Validation Checklist
- [x] Run `python -m tools.benchmark_performance` and capture results under `docs/benchmarks/`.
- [x] Confirm sidebar memory monitor displays usage, top frames, and cleanup controls.
- [x] Verify Elo views load correctly with single-format filters and domestic/live scope toggles.
- [x] Confirm fast ingestion reruns without recomputing when no new files are uploaded.

Maintained by the Performance Working Group. Update this report whenever optimisation milestones land in `master`.
