# Cricket Captain Stat Pack Application

This application allows users to upload Cricket Captain scorecard files (.txt) and view detailed statistics and analysis based on the processed data.

## Application Workflow

1.  **Login:** The application starts with a login screen to authenticate users.
2.  **File Upload:** After logging in, users can upload Cricket Captain scorecard files (.txt).
3.  **Data Processing:** The uploaded files are parsed, and the data is processed to update the application's internal datasets. This includes calculating various batting, bowling, and match statistics.
4.  **View Statistics:** Users can navigate through different views to explore the processed statistics.

## Core Modules

The application's core logic is handled by the following Python modules:

*   **`match.py`**
    *   **Purpose:** Extracts match-level metadata from scorecard `.txt` files.
    *   **Input:** Directory path containing scorecard files.
    *   **Processing:**
        *   Iterates through each `.txt` file.
        *   Parses specific lines (e.g., line 2 for teams, line 3 for competition/date) using string splitting.
        *   Determines `Match_Format` based on keywords in the competition string.
        *   Scans the end of the file for "Man of the match:" and result lines (e.g., "won", "lost", "drawn", "tied").
        *   Uses regex (`re.search`) to extract numeric run/wicket margins from the result string.
        *   Calculates boolean flags for results (`Home_Win`, `Away_Won`, `Tie`, etc.) and an `Innings_Win` flag.
        *   Applies a `transform_competition` function to standardize competition names (e.g., mapping specific team pairings in Test Matches to trophy names like 'The Ashes').
    *   **Output:** A Pandas DataFrame containing one row per match with columns like `File Name`, `Home_Team`, `Away_Team`, `Date`, `Competition`, `Match_Format`, `Player_of_the_Match`, `Match_Result`, `Margin`, result flags, and numeric margins. This DataFrame is also saved to `match_data.csv` in the scorecards directory.
    *   **Dependencies:** `pandas`, `re`, `os`.

*   **`game.py`**
    *   **Purpose:** Extracts innings-level summary statistics from scorecard `.txt` files.
    *   **Input:** Directory path containing scorecard files, `match_df` (DataFrame from `match.py`).
    *   **Processing:**
        *   Iterates through each `.txt` file.
        *   Parses lines to identify innings breaks (`-------------`) and the batting team.
        *   Uses regex (`re.search`) to find the "TOTAL" line for each innings, extracting total runs, wickets, and overs (or balls for 'The Hundred'). Handles different formats of the TOTAL line (e.g., 'all out', 'X wkts').
        *   Calculates `Run_Rate` (Total Runs / Overs) and `Balls_Per_Wicket`.
        *   Merges the extracted innings data with the input `match_df` (on `File Name`) to add match context (`Competition`, `Match_Format`, `Date`, etc.).
    *   **Output:** A Pandas DataFrame containing one row per innings with columns like `File Name`, `Home_Team`, `Away_Team`, `Bat_Team`, `Bowl_Team`, `Innings`, `Total_Runs`, `Overs`, `Wickets`, `Run_Rate`, `Competition`, `Match_Format`, `Date`, etc. This DataFrame is also saved to `game_data.csv` in the scorecards directory.
    *   **Dependencies:** `pandas`, `re`, `os`, `numpy`. Relies on `match_df` from `match.py`.

*   **`bat.py`**
    *   **Purpose:** Extracts detailed batting statistics for each player in each innings from scorecard `.txt` files.
    *   **Input:** Directory path containing scorecard files, `game_df` (DataFrame from `game.py`), `match_df` (DataFrame from `match.py`).
    *   **Processing:**
        *   Iterates through each `.txt` file.
        *   Parses lines following innings separators (`-------------`) to identify batting performances.
        *   Uses a complex regex (`re.search`) to capture `Name`, `How_Out`, `Runs`, `Balls`, `Fours`, `Sixes` for each batter line. Handles 'Did not bat' cases.
        *   Calculates derived statistics:
            *   Boolean flags: `Batted`, `Out`, `Not Out`, `DNB`.
            *   Milestones: `50s`, `100s`, `200s`.
            *   Dismissal types: `Caught`, `Bowled`, `LBW`, `Run Out`, `Stumped`.
            *   Calculated metrics: `Boundary Runs`, `Strike Rate`.
        *   Calculates `Team Balls` based on whether the team was all out or the match format's standard ball count.
        *   Merges with `game_df` (on `File Name`, `Innings`) to add innings context (`Total_Runs`, `Overs`, `Wickets`, `Competition`, `Match_Format`, `Date`).
        *   Applies the same `transform_competition` function as `match.py`.
        *   Extracts `Year` from the `Date`.
    *   **Output:** A Pandas DataFrame containing one row per player per innings batted, with detailed batting stats and merged match/innings context.
    *   **Dependencies:** `pandas`, `re`, `os`, `traceback`. Relies on `game_df` from `game.py` and `match_df` from `match.py`.

*   **`bowl.py`**
    *   **Purpose:** Extracts detailed bowling statistics for each player in each innings from scorecard `.txt` files.
    *   **Input:** Directory path containing scorecard files, `game_df` (DataFrame from `game.py`), `match_df` (DataFrame from `match.py`).
    *   **Processing:**
        *   Iterates through each `.txt` file.
        *   Parses lines following bowling data separators (identified by `Line_no` being 3, 7, 11, or 15 after a `-------------` line).
        *   Uses regex (`re.search`) to capture `Name`, `Overs`, `Maidens`, `Runs`, `Wickets`, `Econ` for each bowler line.
        *   Calculates derived statistics:
            *   Boolean flag: `Bowled`.
            *   Milestones: `5Ws`, `10Ws`.
            *   Calculated metrics: `Bowler_Balls` (handles 5-ball overs for 'The Hundred'), `Runs_Per_Over`, `Balls_Per_Wicket`, `Dot_Ball_Percentage`, `Strike_Rate`, `Average`. Adjusts `Bowler_Overs` and `Bowler_Econ` for 'The Hundred'.
        *   Merges first with `match_df` (on `File Name`) to get `Home_Team`, `Away_Team`, then merges with `game_df` (on `File Name`, `Innings`) to add innings context.
        *   Applies the same `transform_competition` function as `match.py`.
    *   **Output:** A Pandas DataFrame containing one row per player per innings bowled, with detailed bowling stats and merged match/innings context.
    *   **Dependencies:** `pandas`, `re`, `os`, `traceback`. Relies on `game_df` from `game.py` and `match_df` from `match.py`.

*   **`cricketcaptain.py`**
    *   **Purpose:** Main Streamlit application entry point. Handles UI, navigation, authentication, and page loading.
    *   **Processing:**
        *   Sets page configuration (`st.set_page_config`).
        *   Injects custom CSS for styling.
        *   Implements login functionality (`login` function) using `st.text_input` and `st.button`.
        *   Securely retrieves credentials using `get_credentials`, prioritizing Streamlit secrets, then environment variables, then hardcoded values (fallback). Uses `dotenv` optionally for local `.env` files.
        *   Manages session state (`st.session_state`) to track login status (`logged_in`) and timestamp (`login_time`).
        *   Enforces session duration (`is_session_valid`) and displays a timer (`display_session_timer`).
        *   Uses `st.Page` to define individual views/pages located in the `views/` directory.
        *   Uses `st.navigation` to create the sidebar navigation menu.
        *   Adds logos and a logout button to the sidebar.
        *   Calls `pg.run()` to display the currently selected page.
    *   **Dependencies:** `streamlit`, `datetime`, `time`, `os`, optionally `dotenv`.

*   **`database.py`**
    *   **Purpose:** Intended for database interactions.
    *   **Current State:** Empty file. Data persistence is currently handled by saving/loading DataFrames to/from CSV files (`match_data.csv`, `game_data.csv`) generated by `match.py` and `game.py`. The `redis` dependency in `requirements.txt` might suggest planned use for caching or session management, but isn't implemented in these core files.

## Views (Pages)

The application presents data through various views (Streamlit pages located in the `views/` directory):

*   **`Home.py`**: The main landing page after login. Displays an overview, potentially some quick summary statistics, recent updates, or navigation links.
*   **`batview.py`**: Dedicated to batting statistics. Shows detailed batting leaderboards, player career stats, season stats, format-specific stats, opponent stats, location stats, milestone analysis, percentile rankings, and various charts visualizing batting performance (e.g., Avg vs SR).
*   **`bowlview.py`**: Dedicated to bowling statistics. Similar to the batting view, it displays bowling leaderboards, player career stats, season stats, format-specific stats, opponent stats, location stats, and visualizations for bowling performance.
*   **`scorecards.py`**: Allows users to view detailed scorecards for individual matches that have been uploaded.
*   **`recordsview.py`**: Displays all-time records across various categories for both batting (e.g., highest score, most runs) and bowling (e.g., best figures, most wickets).
*   **`allrounders.py`**: Focuses on players who excel in both batting and bowling, likely showing combined rankings or statistics.
*   **`compare.py`**: Allows users to select multiple players and compare their statistics side-by-side.
*   **`domestictables.py`**: (Inferred) Likely displays league tables or standings for domestic competitions.
*   **`elorating.py`**: Implements and displays Elo ratings for teams or players based on match results.
*   **`gamestatsview.py`**: Provides statistics aggregated at the game or match level, possibly showing overall match summaries beyond individual scorecards.
*   **`headtohead.py`**: Shows the statistical record between two specific teams or players when they compete against each other.
*   **`Playerrankings.py`**: Displays overall player rankings, potentially using a custom ranking algorithm or combining various statistical measures.
*   **`rankings.py`**: (Potentially redundant or specific focus) Another view for rankings, perhaps focusing on different criteria or formats compared to `Playerrankings.py`.
*   **`teamview.py`**: Shows statistics aggregated at the team level, such as overall team performance, win/loss records, and team batting/bowling averages.
*   **`versions.py`**: Displays information about the application version, updates, or changelog.
*   **`watch.py`**: (Inferred) Purpose unclear from the name, might be for watching live game updates (if applicable) or tracking specific players/teams.

## Setup and Running

(Add instructions here on how to set up the environment, install dependencies from `requirements.txt`, configure secrets (`setup_secrets.py`), and run the Streamlit application.)

```
Example command:
```bash
pip install -r requirements.txt
streamlit run Home.py
```

*(Please update the Setup and Running section with specific instructions for your project.)*

## Accessing the Dashboard

This application requires a subscription to access. The Cricket Captain Stats Dashboard is a premium tool designed to help you analyze your Cricket Captain game data.

### Login Instructions

1. Subscribe to the "Cricket Captain 2024 Stats Pack tier" on [Buy Me A Coffee - Leading Edge](https://buymeacoffee.com/leadingedgepod)
2. After subscribing, you'll receive the login credentials:

### Using the Dashboard

Once logged in, you can:

1. **Upload your Cricket Captain game data**:
   - In your Cricket Captain game, save the scorecards you want to analyze
   - Find the saved .txt files on your computer:
     - Windows: `C:\Users\[USER NAME]\AppData\Roaming\Childish Things\Cricket Captain 2021`
     - Mac: `~/Library/Containers/com.childishthings.cricketcaptain2021mac/Data/Library/Application Support/Cricket Captain 2021/childish things/cricket captain 2021/saves`
   
2. **Process your data**:
   - Click "Browse files" and select your saved .txt files
   - Select all files using Ctrl+A (Windows) or Command+A (Mac)
   - Click "Process Scorecards" to analyze your data
   - Navigate through the tabs to view different statistics

## Session Information

- Your login session will last for 24 hours before requiring you to log in again
- You can manually log out using the "Logout" button in the sidebar

## Additional Resources

- Watch Cricket Captain 2024 gameplay on [Rob Taylor's YouTube Channel](https://www.youtube.com/@RobTaylor1985)
- Support development through [Buy Me A Coffee](https://buymeacoffee.com/leadingedgepod)

---

*This is a subscription product. Please do not share login credentials.*
````
