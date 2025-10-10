# Cricket Captain Stat Pack

Streamlit-powered analytics suite for Cricket Captain 2025 saves. Upload thousands of scorecards, slice performance by format or competition, and explore rich visual dashboards designed for serious stat-heads.

## ✨ Key Features

- **Mass upload pipeline** – ingest up to 1,200 scorecards at once via the new ZIP importer, or drag-and-drop individual `.txt` files.
- **Smarter competition labelling** – Pakistan Super Cup is correctly recognised as T20, while domestic competitions such as FC League, Super Cup, 20 Over Trophy, and One Day Cup collapse duplicate phase names ("Final", "Semi", etc.).
- **Head-to-head history** – Asia Trophy (T20 and ODI) tournaments now appear in the historical progression matrix, including winners and runners-up.
- **Context-aware Elo ratings** – toggle between *Domestic* and *International* scopes to compare teams in the right competitive arena.
- **Global filters everywhere** – pick one or more formats and competitions and the selection flows across #1 rankings, batting, bowling, and all-rounder dashboards.
- **Modern UI & navigation** – polished cards, gradient themes, and quick links to batting, bowling, team, comparison, records, Elo, and version history views.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or newer
- `pip` (or `pipenv`/`uv` if you prefer)
- A subscription to the **Cricket Captain 2024 Stats Pack** (credentials are required to log in)

### Installation

```powershell
# clone this repository
git clone https://github.com/Chockers1/CricketCaptain2024.git
cd CricketCaptain2024

# (optional) create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows PowerShell

# install dependencies
pip install -r requirements.txt
```

### Configure login credentials

Either create a `.streamlit/secrets.toml` manually or run the helper script and follow the prompts:

```powershell
python setup_secrets.py
```

This generates `.streamlit/secrets.toml` with a `[login]` section. Keep this file out of version control.

### Launch the dashboard

```powershell
streamlit run cricketcaptain.py
```

The app opens in your browser on http://localhost:8501.

## 📥 Uploading Scorecards

1. **Gather scorecards**
   - Windows: `C:\Users\<you>\AppData\Roaming\Childish Things\Cricket Captain 2025\Saves\Scorecards`
   - macOS: `~/Library/Containers/com.childishthings.cricketcaptain2025mac/Data/Library/Application Support/Cricket Captain 2025/childish things/cricket captain 2025/saves`
2. **Choose an import mode**
   - *ZIP mode* (recommended for large archives): compress the folder, upload once, and the app flattens nested directories automatically.
   - *TXT mode*: select individual `.txt` files for smaller batches.
3. Hit **Process Scorecards**. The pipeline reads every file, standardises competition names, stores match/game/batting/bowling extracts, and caches the results in session state.

Progress feedback appears as each stage (match, game, bowling, batting) completes. Duplicate detection, player team switches, and innings overlap checks run automatically.

## 🧭 Navigating the App

- **Home** – load data, review progress summaries, and follow the quick-start guide within the app.
- **Player Rankings** – global format & competition multiselects drive the #1, batting, bowling, and all-rounder tabs. Rankings incorporate refined batting/bowling rating formulas.
- **Elo Ratings** – filter by Format and by the new *Competition Scope* (Domestic vs International) before exploring time-series plots and tables.
- **Head-to-Head** – compare teams with results grids, tournament histories (now including Asia Trophy variants), and recent form cards.
- **Bat View / Bowl View / Team View** – deep dives into player and team production, with aligned column layouts and fixed SR calculations.
- **Records, Scorecards, Compare, Versions** – browse historical records, individual match scorecards, side-by-side player comparisons, and the in-app changelog.

## 🔧 Under the Hood

The ingestion scripts inside the project root orchestrate the data model:

| Module | Description |
| --- | --- |
| `match.py` | Extracts match metadata, detects results/outcomes, and normalises competition names (trophies, league phases, international titles). |
| `game.py` | Captures innings totals, run rates, wickets, overs, and merges match context. |
| `bat.py` | Parses every batting line, calculates milestones, strike rates, dismissal types, and maps competitions using the latest naming rules. |
| `bowl.py` | Processes bowling figures, computes advanced metrics (dot percentage, economy, strike rate), and aligns competition labels with batting/match data. |
| `cricketcaptain.py` | Streamlit entry-point handling authentication, navigation, and session lifetime. |

Processed DataFrames (`match_data.csv`, `game_data.csv`, `bat_df`, `bowl_df`) persist in memory during a session and can be exported if you extend the app.

## 🔐 Access & Sessions

- Login is required for every browser session. Credentials are supplied to subscribers via [Buy Me A Coffee – Leading Edge](https://buymeacoffee.com/leadingedgepod).
- Sessions remain active for 24 hours or until you choose **Logout** from the sidebar.

## 🙌 Supporting Resources

- YouTube: [Rob Taylor](https://www.youtube.com/@RobTaylor1985)
- Support the project: [Buy Me A Coffee](https://buymeacoffee.com/leadingedgepod)
- Release notes live inside the app on the **Versions** tab (v1.24 details the ZIP importer, Asia Trophy coverage, Elo scope filter, and competition clean-up).

---

*The Cricket Captain Stat Pack is a subscription product. Please keep your credentials private.*
````
