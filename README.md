# Cricket Captain Stat Pack

Streamlit-powered analytics suite for Cricket Captain 2025 saves. Upload thousands of scorecards, slice performance by format or competition, and explore rich visual dashboards designed for serious stat-heads.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or newer
- `pip` (or `pipenv`/`uv` if you prefer)
- A subscription to the **Cricket Captain 2024 Stats Pack** (credentials are required to log in)

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

### 🏠 **Home**
Data upload hub with ZIP/TXT file processing and progress summaries. Features quick-start guide and helpful resource links.

### 🏏 **Batting View** (`batview.py`)
Comprehensive batting analytics across multiple dimensions:
- **Stats Used:** Runs, balls faced, dismissals, boundaries, milestones (50s, 100s, 200s), strike rates
- **Analysis Types:** 
  - Career stats (lifetime aggregated performance)
  - Season-by-season breakdowns
  - Opposition-specific performance 
  - Location/venue analysis (home vs away)
  - Batting position trends
  - Form tracking (recent innings performance)
- **Calculations:** Batting averages, strike rates, balls per dismissal, runs per match, milestone frequencies, dismissal patterns

### 🎳 **Bowling View** (`bowlview.py`)
Detailed bowling performance analysis:
- **Stats Used:** Overs, runs conceded, wickets, maidens, economy rates, bowling figures
- **Analysis Types:**
  - Career bowling statistics
  - Season performance trends
  - Opposition-specific effectiveness
  - Location-based analysis
  - Bowling position/role analysis
- **Calculations:** Bowling averages, economy rates, strike rates, dot ball percentages, 5-wicket and 10-wicket hauls, wickets per match

### 🚀 **All Rounders** (`allrounders.py`)
Combined batting and bowling performance for multi-skilled players:
- **Stats Used:** Merged batting and bowling statistics with weighted performance ratings
- **Analysis Types:**
  - Career all-rounder stats
  - Season-based dual-discipline tracking
  - Opposition effectiveness in both disciplines
- **Calculations:** Combined batting/bowling ratings, all-rounder rankings, dual-discipline performance indices

### 🏆 **Team View** (`teamview.py`)
Team-level performance analytics and comparisons:
- **Stats Used:** Aggregated team batting/bowling statistics, match results, performance indices
- **Analysis Types:**
  - Team career statistics (batting and bowling combined)
  - Season performance comparisons
  - Opposition-specific team records
  - Location-based team performance
  - Team rankings and performance indices
- **Calculations:** Team batting/bowling averages, performance indices (weighted against format means), batting vs bowling average differentials, team rankings

### ⚖️ **Compare Players** (`compare.py`)
Head-to-head player comparisons:
- **Stats Used:** Complete batting and bowling career statistics for selected players
- **Analysis Types:**
  - Side-by-side statistical comparisons
  - Category-based scoring (batting performance, milestones, bowling effectiveness)
  - Year-over-year performance tracking
- **Calculations:** Performance differentials, category winners, career trajectory analysis

### � **Similar Players** (`views/similarplayers.py`)
Discover batting and bowling doppelgängers using dual-mode similarity engines:
- **Core Features:** Parallel batting/bowling tabs, tolerance filters, distance-weighted comparisons, match-format and team filters, position-specific slicing
- **Visualisations:** Scatter similarity maps, radar skill profiles, metric-difference bar charts, correlation heatmaps for both disciplines
- **Calculations:** Weighted Euclidean distances, tolerance-matched cohorts, normalized metric scoring, similarity percentages with interactive progress columns
- **Use Cases:** Talent scouting, role-based replacements, player archetype searches across formats and eras

### �📈 **Player Rankings** (`Playerrankings.py`)
Comprehensive player ranking system:
- **Stats Used:** Advanced batting and bowling rating formulas incorporating performance context
- **Analysis Types:**
  - Global player rankings across formats
  - Batting-specific rankings
  - Bowling-specific rankings  
  - All-rounder rankings (combined discipline ratings)
- **Calculations:** Weighted batting/bowling ratings, format-specific rankings, performance-based scoring systems

### ♟️ **Elo Ratings** (`elorating.py`)
Dynamic team strength ratings based on match results:
- **Stats Used:** Match results, team performance over time, competition context
- **Analysis Types:**
  - Time-series Elo rating evolution
  - Format-specific ratings (Test, ODI, T20I)
  - Competition scope filtering (Domestic vs International)
- **Calculations:** Elo rating system with K-factor adjustments, expected match outcomes, rating changes based on results

### 🆚 **Head-to-Head** (`headtohead.py`)
Team vs team historical analysis:
- **Stats Used:** Match results, series records, recent form, tournament histories
- **Analysis Types:**
  - Results grids and win/loss records
  - Tournament histories including Asia Trophy variants
  - Recent form cards and trends
- **Calculations:** Win percentages, series statistics, form trends, head-to-head records

### 📜 **Records** (`recordsview.py`)
Historical records and milestone achievements:
- **Stats Used:** Individual batting/bowling performances, match figures, series statistics
- **Analysis Types:**
  - Best batting performances (highest scores, partnerships)
  - Best bowling figures (5-wicket hauls, best match figures)
  - Series records and tournament achievements
- **Calculations:** Record identification, ranking of performances, milestone tracking

### 📜 **Scorecards** (`scorecards.py`)
Individual match scorecard viewer:
- **Stats Used:** Complete match data including innings details, batting/bowling figures
- **Analysis Types:** Detailed match recreations with full scorecards
- **Calculations:** Match summaries, innings totals, individual contributions

### 📖 **Versions** (`versions.py`)
In-app changelog and version history detailing feature updates and improvements.

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
