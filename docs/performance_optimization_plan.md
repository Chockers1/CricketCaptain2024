# Performance Optimization Plan

## Executive Summary
- Fast ingestion is now the default path for every upload and delivers sub-second parsing on the 3,208-scorecard benchmark.
- The memory management layer (cache-aware DataFrame registry, auto dtype coercion, session cleanup) is rolled out across every view.
- All Elo tables, rankings, and charts now render from a single cached calculation and survive aggressive filter combinations without errors.
- Remaining work focuses on predictable cache invalidation, Polars coverage for heavy aggregations, and automated regression benchmarks.

## Current Health Snapshot (Oct 2025)
| Area | Status | Notes |
|------|--------|-------|
| Ingestion pipeline | ✅ Stable | `FastCricketProcessor` front-loads parsing, legacy processors retained as fallback. |
| Session memory | ✅ Stable | `PerformanceManager` tracks lifetimes, sidebar monitor available in all primary views. |
| View responsiveness | ✅ Stable | Bat/Bowl/Records/Ranking views hydrate from cached Polars frames; Elo matrices reworked for filter safety. |
| Documentation | ⚠️ Needs polish | README refreshed, but deeper contributor docs captured in this plan and the implementation playbook. |
| Automated assurance | ⚠️ In progress | Smoke script exists; formal pytest suite and benchmark workflow still to-do. |

## Completed Initiatives
1. **Fast Mode Everywhere**
   - Ingestion defaults to `FastCricketProcessor` (Streamlit Home view, CLI utilities, tests).
   - Bowling/Batting processors share regex caches and sanitisation helpers.
   - Benchmark harness moved to `tools/benchmark_performance.py` for repeatable timing checks.
2. **Memory Management Pass**
   - Removed redundant `.copy()` usage across all views.
   - Added sidebar telemetry and one-click cleanup using `PerformanceManager`.
   - Implemented type downcasting and categorical normalisation during ingestion.
3. **Elo Resilience Work**
   - Centralised Elo cache with `data_types=['match']` tagging for fast invalidation.
   - Ranking matrix rebuilt from the full dataset; filter selections highlight rather than remove teams.
   - Progression and distribution charts hardened for empty-data cases.
4. **Documentation Cleanup**
   - `docs/` folder contains this plan, the implementation guide, and the optimisation status report.
   - README summarises new project layout, workflows, and troubleshooting steps.

## Near-Term Priorities (Q4 2025)
1. **Cache Hygiene**
   - [ ] Ship hash-based cache busting in `memory_efficient_cache` (invalidate on file digest, not time).
   - [ ] Add cache usage metrics to the sidebar monitor (hit/miss counters by data type).
2. **Analytics on Polars**
   - [ ] Port Bat and Bowl ranking aggregations to Polars for multi-format queries.
   - [ ] Replace Records streak calculations with Polars window functions.
3. **Quality Gates**
   - [ ] Convert the fast-processing smoke check into `pytest` under `tests/` with sample fixtures.
   - [ ] Add GitHub workflow or local script to run `python -m tools.benchmark_performance` and fail on regressions beyond tolerance.
   - [ ] Introduce linting (ruff or flake8) and type checking (mypy) pre-commit hooks.
4. **User Experience Polish**
   - [ ] Add lazy-loading indicators for heavy tabs (Records, Elo History).
   - [ ] Surface cache age and source dataset metadata on every view.
   - [ ] Expand dark-mode-friendly styling.

## Longer-Term Opportunities
- **Data Warehouse Hand-off:** Persist processed scorecards to Parquet in `/data/warehouse/` for downstream analytics.
- **Model Serving:** Explore deploying fast ingestion as an API for batch updates without opening the Streamlit UI.
- **Scenario Simulation:** Build a lightweight engine on top of cached stats to simulate fixtures and tournament scenarios.
- **Client Distribution:** Package the Streamlit app with install scripts so non-technical users can run locally.

## Work Streams & Owners
| Stream | Owner | Inputs | Deliverables |
|--------|-------|--------|--------------|
| Cache lifecycle | Performance Team | `performance_utils`, `memory_efficient_cache` | Hash-based invalidation, monitoring UI |
| Polars port | Analytics Team | `fast_processing`, view aggregations | Polars DataFrame utilities, regression benchmarks |
| Quality gates | Tooling Team | `tools/`, `tests/` | Automated benchmarks, pytest suite, CI plan |
| UX polish | Front-end Team | `views/`, assets | Loading states, theming, metadata widgets |

## Milestones
- **Nov 15** – Cache metrics in sidebar; regression benchmark recorded and saved to `docs/benchmarks/`.
- **Dec 01** – Bat/Bowl aggregations fully on Polars with parity tests.
- **Dec 15** – CI pipeline running benchmarks + pytest; README includes contributor workflow.
- **Jan 05** – UX polish delivered; release notes captured in `docs/changelog.md`.

## Reference Materials
- `docs/performance_implementation_guide.md` – step-by-step checklist for applying optimisations.
- `docs/optimization_status_report.md` – running log of completed performance work.
- `tools/benchmark_performance.py` – command-line benchmark harness.

---
Maintained by the Performance Working Group. Update this plan after each sprint review or whenever scope changes materially.
