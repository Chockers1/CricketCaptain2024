# Cricket Captain Stat Pack

Streamlit-powered performance analytics for Cricket Captain 2024/2025 career saves. The Stat Pack ingests thousands of scorecards, builds competition-aware metrics, and surfaces polished dashboards for serious analysis.

---

## What It Does
- Bulk-ingest ZIP or TXT scorecards with a Polars-first pipeline optimised for speed and memory.
- Auto-standardise competitions, formats, and player identities so every view is filter-safe.
- Serve interactive Streamlit dashboards covering batting, bowling, Elo ratings, records, head-to-head, and more.
- Track every release inside the app via the Versions view (`views/versions.py`), including regression checklists and benchmarks.

## How It Works
- **Ingestion** – `fast_processing.py` powers the FastCricketProcessor, flattening uploads into `bat`, `bowl`, `match`, and `game` tables while logging `[FAST]` timing telemetry.
- **Analytics layer** – Each dashboard in `views/` reads shared cached DataFrames, applying lightweight transforms for its own layout.
- **Caching & memory** – `performance_utils.py` and `memory_efficient_cache` decorators avoid redundant copies and expose sidebar memory controls.
- **Release governance** – `views/versions.py` centralises the changelog, pulls optional benchmarks from `tools/benchmark_performance`, and displays regression notes directly inside the app.

## Quick Start
**Prerequisites**
- Python 3.10+
- `pip`
- Active Cricket Captain Stat Pack subscription (credentials required at login)

**Install & run**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run cricketcaptain.py
```

## Operational Checklist
1. `python -m tools.fast_processing_diagnostics` – validate regex wiring and ingestion helpers.
2. `python -m tools.benchmark_performance` – compare fast vs legacy pipeline timings (log the results under `docs/benchmarks/`).
3. Launch Streamlit, upload a ZIP of scorecards, and confirm `[FAST]` timings plus tab-level cache hits in the console.

## Project Layout
```
├── bat.py / bowl.py / match.py / game.py  # Core ingestion modules
├── cricketcaptain.py                      # Streamlit entry point
├── fast_processing.py                     # FastCricketProcessor + helpers
├── performance_utils.py                   # Memory + caching utilities
├── views/                                 # Streamlit page definitions (incl. Versions changelog)
├── tools/                                 # CLI utilities for benchmarks and diagnostics
├── docs/                                  # Optimisation plan, implementation guide, status reports
├── examples/                              # Reference snippets and helper utilities
└── data/sample_scorecards/                # Anonymised sample scorecards for local testing
```

Key documentation lives in `docs/`:
- `performance_optimization_plan.md` – roadmap, priorities, and upcoming experiments.
- `performance_implementation_guide.md` – step-by-step optimisation checklist.
- `optimization_status_report.md` – rolling summary of completed work and open items.

## Release Management
- Latest release: **v1.26 (2025-10-31)** – ship the Polars-powered Fast Mode pipeline, speeding end-to-end ingest by 60–75% and wiring benchmarks into the Versions tab.
- The Versions Streamlit tab reads directly from `views/versions.py`. Add new releases here (date, title, narrative, optional screenshots) and document regression checks.
- Use the benchmark expander in the Versions tab to capture ingest timings before shipping.

## Running the App
1. Collect scorecards:
   - Windows: `C:\Users\<you>\AppData\Roaming\Childish Things\Cricket Captain 2025\Saves\Scorecards`
   - macOS: `~/Library/Containers/com.childishthings.cricketcaptain2025mac/Data/Library/Application Support/Cricket Captain 2025/childish things/cricket captain 2025/saves`
2. Choose an import mode:
   - **ZIP mode** (recommended) – drag entire archives; the app flattens and deduplicates automatically.
   - **TXT mode** – pick individual scorecards for quick validation.
3. Click **Process Scorecards**. The pipeline normalises competitions, hydrates aggregated tables, and caches them for instant reuse across tabs.

Navigation highlights:
- **Home** – upload wizard, progress indicator, ingest telemetry.
- **Bat/Bowl/All-Rounders** – Polars-backed analytics with memory monitors and export-ready tables.
- **Compare & Similar Players** – deep scouting tools including tolerance/distance weighting.
- **Head-to-Head & Records** – historical matrices, tournament winners, and percentiles.
- **Versions** – curated changelog, regression checklist, and optional screenshots per release.

## Data & Samples
- `data/sample_scorecards/` ships with anonymised examples for tests and demos.
- Re-uploading the same archive rehydrates from cache instantly; clear session state from the sidebar to force a clean ingest.
- Extend `fast_processing.py` if you need to export processed tables to Parquet or other downstream formats.

## Support & Contact
- YouTube: [Rob Taylor](https://www.youtube.com/@RobTaylor1985)
- Support: [Buy Me A Coffee – Leading Edge](https://buymeacoffee.com/leadingedgepod)
- Release notes: available in-app on the **Versions** tab.

---

_The Cricket Captain Stat Pack is a subscription product. Keep your credentials private and do not distribute processed data without permission._
````
