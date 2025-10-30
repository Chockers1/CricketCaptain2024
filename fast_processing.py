"""Fast scorecard processing helpers for bulk scorecard ingestion."""

from __future__ import annotations

import multiprocessing
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import polars as pl

from game import process_game_stats
from match import process_match_data


PLAYER_PATTERN = re.compile(
    r"^(?P<Name>.+?)(?:\s+(?P<How_Out>(lbw|c|b|not out|run out|st|retired).+?))?\s+"
    r"(?P<Runs>\d+)\s+(?P<Balls>\d+)\s+(?P<Fours>\d+|-)\s+(?P<Sixes>\d+|-)$"
)

BOWLER_PATTERN = re.compile(
    r"^(?P<Name>.+?)\s+(?P<Overs>\d+(?:\.\d+)?)\s+(?P<Maidens>\d+)\s+"
    r"(?P<Runs>\d+)\s+(?P<Wickets>\d+)(?:\s+(?P<Econ>\d+\.\d+))?$"
)


def parse_overs_to_balls(overs_expr: pl.Expr) -> pl.Expr:
    """Convert a Polars expression of overs (e.g., 45.5) into total balls."""

    return (overs_expr.floor().cast(pl.Int64) * 6) + (
        ((overs_expr - overs_expr.floor()) * 10).round(0).cast(pl.Int64)
    )


class FastCricketProcessor:
    """High-performance scorecard ingestion with cached parsing."""

    def __init__(self, use_multiprocessing: bool = True, max_workers: Optional[int] = None) -> None:
        self.use_multiprocessing = use_multiprocessing
        cpu_count = multiprocessing.cpu_count() or 1
        self.max_workers = max_workers or min(8, cpu_count)

    @lru_cache(maxsize=1000)
    def cached_date_parse(self, date_str: str) -> pd.Timestamp:
        """Cache parsed dates to avoid repeated parsing of identical values."""

        try:
            return pd.to_datetime(date_str, format="%d %b %Y")
        except Exception:
            return pd.to_datetime(date_str, dayfirst=True, errors="coerce")

    def _parse_scorecard(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Read a scorecard file and capture the canonical metadata."""

        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            return None

        lines = content.splitlines()
        home_team = ""
        away_team = ""
        if len(lines) >= 2 and " v " in lines[1]:
            home_team, away_team = [part.strip() for part in lines[1].split(" v ", 1)]

        return {
            "filename": file_path.name,
            "lines": lines,
            "home_team": home_team,
            "away_team": away_team,
        }

    def process_file_batch(
        self, file_paths: Sequence[Path], batch_size: int = 50
    ) -> List[Dict[str, Any]]:
        """Load scorecard text in moderately sized batches."""

        if not file_paths:
            return []

        paths = list(file_paths)
        results: List[Dict[str, Any]] = []

        for start in range(0, len(paths), batch_size):
            batch = paths[start : start + batch_size]
            if self.use_multiprocessing and self.max_workers > 1:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    for parsed in executor.map(self._parse_scorecard, batch):
                        if parsed:
                            results.append(parsed)
            else:
                for file_path in batch:
                    parsed = self._parse_scorecard(file_path)
                    if parsed:
                        results.append(parsed)

        return results

    def fast_batting_extraction(self, processed_files: Iterable[Dict[str, Any]]) -> pd.DataFrame:
        """Extract batting records from parsed scorecards."""

        records: List[List[Any]] = []

        for file_data in processed_files:
            if not file_data:
                continue

            lines = file_data["lines"]
            filename = file_data["filename"]
            home_team = file_data["home_team"]
            away_team = file_data["away_team"]

            innings_index = 0
            separator_count = 0

            for idx, raw_line in enumerate(lines):
                if "-------------" not in raw_line:
                    continue

                separator_count += 1
                if separator_count not in (1, 5, 9, 13):
                    continue

                innings_index += 1
                bat_team = raw_line.split("-", 1)[0].strip()
                bowl_team = away_team if bat_team == home_team else home_team

                position = 1
                scan_idx = idx + 1

                while scan_idx < len(lines) and position <= 11:
                    player_line = lines[scan_idx].strip()
                    scan_idx += 1

                    if not player_line:
                        continue
                    if player_line.startswith("Extras") or player_line.startswith("Total"):
                        break

                    match = PLAYER_PATTERN.search(player_line)
                    if match:
                        name = match.group("Name").replace("rtrd ht", "").strip()
                        how_out = match.group("How_Out")
                        runs = int(match.group("Runs"))
                        balls = int(match.group("Balls"))
                        fours = 0 if match.group("Fours") == "-" else int(match.group("Fours"))
                        sixes = 0 if match.group("Sixes") == "-" else int(match.group("Sixes"))
                    else:
                        name = player_line
                        how_out = "Did not bat"
                        runs = balls = fours = sixes = 0

                    records.append(
                        [
                            filename,
                            innings_index,
                            bat_team,
                            bowl_team,
                            position,
                            name,
                            how_out,
                            runs,
                            balls,
                            fours,
                            sixes,
                            home_team,
                            away_team,
                        ]
                    )
                    position += 1

                while position <= 11:
                    records.append(
                        [
                            filename,
                            innings_index,
                            bat_team,
                            bowl_team,
                            position,
                            "",
                            "Did not bat",
                            0,
                            0,
                            0,
                            0,
                            home_team,
                            away_team,
                        ]
                    )
                    position += 1

        columns = [
            "File Name",
            "Innings",
            "Bat Team",
            "Bowl Team",
            "Position",
            "Name",
            "How Out",
            "Runs",
            "Balls",
            "4s",
            "6s",
            "Home Team",
            "Away Team",
        ]

        if not records:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(records, columns=columns)

    def fast_bowling_extraction(self, processed_files: Iterable[Dict[str, Any]]) -> pd.DataFrame:
        """Extract bowling records from parsed scorecards."""

        records: List[List[Any]] = []

        for file_data in processed_files:
            if not file_data:
                continue

            lines = file_data["lines"]
            filename = file_data["filename"]
            home_team = file_data["home_team"]
            away_team = file_data["away_team"]

            innings_index = 0
            separator_count = 0

            for idx, raw_line in enumerate(lines):
                if "-------------" not in raw_line:
                    continue

                separator_count += 1
                if separator_count not in (3, 7, 11, 15):
                    continue

                innings_index += 1
                bat_team = raw_line.split("-", 1)[0].strip()
                bowl_team = away_team if bat_team == home_team else home_team
                position = 1

                for offset in range(1, 12):
                    bowler_idx = idx + offset
                    if bowler_idx >= len(lines):
                        break

                    bowler_line = lines[bowler_idx].strip()
                    if not bowler_line:
                        continue

                    match = BOWLER_PATTERN.search(bowler_line)
                    if not match:
                        continue

                    name = match.group("Name").strip()
                    overs = float(match.group("Overs"))
                    maidens = int(match.group("Maidens"))
                    runs = int(match.group("Runs"))
                    wickets = int(match.group("Wickets"))
                    econ_value = match.group("Econ")
                    economy = float(econ_value) if econ_value else (runs / overs if overs > 0 else 0.0)

                    records.append(
                        [
                            filename,
                            innings_index,
                            bat_team,
                            bowl_team,
                            position,
                            name,
                            overs,
                            maidens,
                            runs,
                            wickets,
                            economy,
                        ]
                    )
                    position += 1

                while position <= 11:
                    records.append(
                        [
                            filename,
                            innings_index,
                            bat_team,
                            bowl_team,
                            position,
                            "",
                            0.0,
                            0,
                            0,
                            0,
                            0.0,
                        ]
                    )
                    position += 1

        columns = [
            "File Name",
            "Innings",
            "Bat Team",
            "Bowl Team",
            "Position",
            "Name",
            "Bowler_Overs",
            "Maidens",
            "Bowler_Runs",
            "Bowler_Wkts",
            "Bowler_Econ",
        ]

        if not records:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(records, columns=columns)

    def process_all_scorecards(self, directory_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process match, bowling, and batting stats using the fast pipeline."""

        directory = Path(directory_path)
        txt_files = sorted(directory.glob("*.txt"))
        processed_files = self.process_file_batch(txt_files)

        match_df = process_match_data(directory_path)
        game_df = process_game_stats(directory_path, match_df)

        bat_df = fast_process_bat_stats(
            directory_path,
            game_df,
            match_df,
            processed_files=processed_files,
            processor=self,
        )

        bowl_df = fast_process_bowl_stats(
            directory_path,
            game_df,
            match_df,
            processed_files=processed_files,
            processor=self,
        )

        return game_df, bowl_df, bat_df


def _normalize_competition(expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(expr.str.contains("Test Match", literal=False))
        .then(pl.lit("Test Match"))
        .when(expr.str.starts_with("World Cup 20"))
        .then(pl.lit("T20 World Cup"))
        .when(expr.str.starts_with("World Cup"))
        .then(pl.lit("ODI World Cup"))
        .when(expr.str.starts_with("Champions Cup"))
        .then(pl.lit("Champions Cup"))
        .when(expr.str.contains("One Day International"))
        .then(pl.lit("ODI"))
        .when(expr.str.contains("20 Over International"))
        .then(pl.lit("T20I"))
        .when(expr.str.contains("Australian League"))
        .then(pl.lit("Sheffield Shield"))
        .when(expr.str.contains("English FC League - D2"))
        .then(pl.lit("County Championship Division 2"))
        .when(expr.str.contains("English FC League - D1"))
        .then(pl.lit("County Championship Division 1"))
        .when(expr.str.contains("Challenge Trophy"))
        .then(pl.lit("Royal London Cup"))
        .otherwise(expr)
    )


def _coalesce_column(df: pd.DataFrame, base_name: str) -> None:
    """Ensure merged DataFrames expose a single column without _x/_y suffixes."""

    candidates = [base_name, f"{base_name}_y", f"{base_name}_x"]
    sources: List[pd.Series] = []

    for name in candidates:
        if name in df.columns:
            series = df[name].astype("string")
            series = series.replace("", pd.NA)
            sources.append(series)

    if not sources:
        return

    combined = sources[0]
    for series in sources[1:]:
        combined = combined.combine_first(series)

    df[base_name] = combined.fillna("")

    for name in [f"{base_name}_x", f"{base_name}_y"]:
        if name in df.columns:
            df.drop(columns=[name], inplace=True)


def fast_process_bat_stats(
    directory_path: str,
    game_df: pd.DataFrame,
    match_df: pd.DataFrame,
    *,
    processed_files: Optional[Sequence[Dict[str, Any]]] = None,
    processor: Optional[FastCricketProcessor] = None,
) -> pd.DataFrame:
    """Vectorised batting processor that mirrors process_bat_stats."""

    print("🚀 Using FAST batting processor...")
    start_time = time.perf_counter()

    processor = processor or FastCricketProcessor()
    file_cache = processed_files

    if file_cache is None:
        txt_files = sorted(Path(directory_path).glob("*.txt"))
        print(f"Found {len(txt_files)} files to process")
        batch_start = time.perf_counter()
        file_cache = processor.process_file_batch(txt_files)
        print(f"🗃️ Processed file contents in {time.perf_counter() - batch_start:.2f} seconds")
    else:
        print(f"🗃️ Reusing pre-processed scorecards ({len(file_cache)} files)")

    extraction_start = time.perf_counter()
    bat_df = processor.fast_batting_extraction(file_cache)
    print(f"📥 Extracted batting data in {time.perf_counter() - extraction_start:.2f} seconds")

    if bat_df.empty:
        print("⚠️ No batting records extracted")
        return bat_df

    bat_df = bat_df.rename(columns={"Bat Team": "Bat_Team", "Bowl Team": "Bowl_Team"})

    if game_df is not None and not game_df.empty:
        merge_start = time.perf_counter()
        merge_cols = [
            "File Name",
            "Innings",
            "Bat_Team",
            "Bowl_Team",
            "Total_Runs",
            "Overs",
            "Wickets",
            "Competition",
            "Match_Format",
            "Player_of_the_Match",
            "Date",
        ]
        merge_cols = [col for col in merge_cols if col in game_df.columns]

        merged_df = bat_df.merge(
            game_df[merge_cols],
            on=["File Name", "Innings"],
            how="left",
        )

        merged_pl = pl.from_pandas(merged_df)

        merged_pl = merged_pl.with_columns(
            pl.when(pl.col("Wickets") == 10)
            .then(pl.col("Balls").sum().over(["File Name", "Innings"]))
            .otherwise(
                pl.when(pl.col("Match_Format").is_in(["The Hundred", "100 Ball Trophy"]))
                .then(100)
                .when(pl.col("Match_Format") == "T20")
                .then(120)
                .when(pl.col("Match_Format") == "One Day")
                .then(300)
                .otherwise(parse_overs_to_balls(pl.col("Overs")))
            )
            .alias("Team Balls")
        )

        merged_pl = merged_pl.with_columns(
            [
                pl.col("Date").str.slice(-4).cast(pl.Int64, strict=False).cast(pl.Utf8).fill_null("0").alias("Year"),
                (pl.col("How Out").fill_null("") != "Did not bat").cast(pl.Int64).alias("Batted"),
                (
                    (pl.col("How Out").fill_null("") != "Did not bat")
                    & (pl.col("How Out").fill_null("") != "not out")
                )
                .cast(pl.Int64)
                .alias("Out"),
                (pl.col("How Out").fill_null("") == "not out").cast(pl.Int64).alias("Not Out"),
                (pl.col("How Out").fill_null("") == "Did not bat").cast(pl.Int64).alias("DNB"),
                ((pl.col("Runs") >= 50) & (pl.col("Runs") < 100)).cast(pl.Int64).alias("50s"),
                ((pl.col("Runs") >= 100) & (pl.col("Runs") < 200)).cast(pl.Int64).alias("100s"),
                (pl.col("Runs") >= 200).cast(pl.Int64).alias("200s"),
                (
                    (pl.col("Runs") <= 25)
                    & (pl.col("How Out").fill_null("") != "Did not bat")
                    & (pl.col("How Out").fill_null("") != "not out")
                )
                .cast(pl.Int64)
                .alias("<25&Out"),
                pl.col("How Out").fill_null("").str.starts_with("c ").cast(pl.Int64).alias("Caught"),
                pl.col("How Out").fill_null("").str.starts_with("b ").cast(pl.Int64).alias("Bowled"),
                pl.col("How Out").fill_null("").str.starts_with("lbw ").cast(pl.Int64).alias("LBW"),
                pl.col("How Out").fill_null("").str.starts_with("run").cast(pl.Int64).alias("Run Out"),
                pl.col("How Out").fill_null("").str.starts_with("st").cast(pl.Int64).alias("Stumped"),
                ((pl.col("4s") * 4) + (pl.col("6s") * 6)).alias("Boundary Runs"),
                pl.when(pl.col("Balls") > 0)
                .then((pl.col("Runs") / pl.col("Balls") * 100).round(2))
                .otherwise(0)
                .alias("Strike Rate"),
                _normalize_competition(pl.col("Competition")).alias("Competition"),
            ]
        )

        bat_df = merged_pl.to_pandas()
        bat_df = bat_df[bat_df["Name"].astype(str).str.strip() != ""]

        print(
            f"🔗 Batting merge and calculations completed in {time.perf_counter() - merge_start:.2f} seconds"
        )

    _coalesce_column(bat_df, "Bat_Team")
    _coalesce_column(bat_df, "Bowl_Team")

    for column in ["Bat_Team", "Bowl_Team", "Home Team", "Away Team", "Player_of_the_Match"]:
        if column in bat_df.columns:
            bat_df[column] = bat_df[column].fillna("").astype(str)

    if "Year" in bat_df.columns:
        bat_df["Year"] = bat_df["Year"].fillna("0").astype(str)

    total_time = time.perf_counter() - start_time
    print(
        f"✅ Fast batting processing completed in {total_time:.2f} seconds"
        f" (records={len(bat_df)}, files={len(file_cache)})"
    )

    return bat_df


def fast_process_bowl_stats(
    directory_path: str,
    game_df: pd.DataFrame,
    match_df: pd.DataFrame,
    *,
    processed_files: Optional[Sequence[Dict[str, Any]]] = None,
    processor: Optional[FastCricketProcessor] = None,
) -> pd.DataFrame:
    """Vectorised bowling processor mirroring process_bowl_stats."""

    print("🚀 Using FAST bowling processor...")
    start_time = time.perf_counter()

    processor = processor or FastCricketProcessor()
    file_cache = processed_files

    if file_cache is None:
        txt_files = sorted(Path(directory_path).glob("*.txt"))
        print(f"Found {len(txt_files)} files to process")
        batch_start = time.perf_counter()
        file_cache = processor.process_file_batch(txt_files)
        print(f"🗃️ Processed file contents in {time.perf_counter() - batch_start:.2f} seconds")
    else:
        print(f"🗃️ Reusing pre-processed scorecards ({len(file_cache)} files)")

    extraction_start = time.perf_counter()
    bowl_df = processor.fast_bowling_extraction(file_cache)
    print(f"📥 Extracted bowling data in {time.perf_counter() - extraction_start:.2f} seconds")

    if bowl_df.empty:
        print("⚠️ No bowling records extracted")
        return bowl_df

    if match_df is not None and not match_df.empty:
        merge_cols = [col for col in ["File Name", "Home_Team", "Away_Team"] if col in match_df.columns]
        if merge_cols:
            bowl_df = bowl_df.merge(match_df[merge_cols], on="File Name", how="left")

    if {"Bat Team", "Bowl Team"}.issubset(bowl_df.columns):
        bowl_df = bowl_df.rename(columns={"Bat Team": "Bat_Team", "Bowl Team": "Bowl_Team"})

    if game_df is not None and not game_df.empty:
        merge_start = time.perf_counter()

        merge_cols = [
            "File Name",
            "Innings",
            "Bat_Team",
            "Bowl_Team",
            "Total_Runs",
            "Overs",
            "Wickets",
            "Competition",
            "Match_Format",
            "Player_of_the_Match",
            "Date",
        ]
        merge_cols = [col for col in merge_cols if col in game_df.columns]

        merged_df = bowl_df.merge(
            game_df[merge_cols],
            on=["File Name", "Innings"],
            how="left",
        )

        merged_pl = pl.from_pandas(merged_df)

        merged_pl = merged_pl.with_columns(
            [
                pl.col("Bowler_Overs").alias("Bowler_Overs_Orig"),
                pl.col("Bowler_Econ").alias("Bowler_Econ_Orig"),
            ]
        )

        merged_pl = merged_pl.with_columns(
            pl.when(pl.col("Match_Format").is_in(["The Hundred", "100 Ball Trophy"]))
            .then(pl.col("Bowler_Overs_Orig"))
            .otherwise(parse_overs_to_balls(pl.col("Bowler_Overs_Orig")))
            .alias("Bowler_Balls")
        )

        merged_pl = merged_pl.with_columns(
            [
                pl.when(pl.col("Match_Format").is_in(["The Hundred", "100 Ball Trophy"]))
                .then(pl.col("Bowler_Balls") / 5)
                .otherwise(pl.col("Bowler_Overs_Orig"))
                .alias("Bowler_Overs"),
                pl.when(pl.col("Match_Format").is_in(["The Hundred", "100 Ball Trophy"]))
                .then(
                    pl.when(pl.col("Bowler_Balls") > 0)
                    .then((pl.col("Bowler_Runs") / pl.col("Bowler_Balls")) * 5)
                    .otherwise(0.0)
                )
                .otherwise(pl.col("Bowler_Econ_Orig"))
                .alias("Bowler_Econ"),
            ]
        )

        merged_pl = merged_pl.with_columns(
            [
                pl.lit(1).alias("Bowled"),
                ((pl.col("Bowler_Wkts") >= 5) & (pl.col("Bowler_Wkts") < 10))
                .cast(pl.Int64)
                .alias("5Ws"),
                (pl.col("Bowler_Wkts") >= 10).cast(pl.Int64).alias("10Ws"),
                pl.col("Date").str.slice(-4).cast(pl.Int64, strict=False).cast(pl.Utf8).fill_null("0").alias("Year"),
                _normalize_competition(pl.col("Competition")).alias("Competition"),
            ]
        )

        merged_pl = merged_pl.with_columns(
            [
                (
                    pl.col("Bowler_Runs").cast(pl.Float64)
                    / pl.when(pl.col("Bowler_Overs") > 0)
                    .then(pl.col("Bowler_Overs"))
                    .otherwise(None)
                )
                .fill_null(0)
                .alias("Runs_Per_Over"),
                pl.when(pl.col("Bowler_Wkts") > 0)
                .then((pl.col("Bowler_Balls") / pl.col("Bowler_Wkts")).round(2))
                .otherwise(pl.col("Bowler_Balls"))
                .alias("Balls_Per_Wicket"),
                (
                    (pl.col("Maidens") * 6).cast(pl.Float64)
                    / pl.when(pl.col("Bowler_Balls") > 0)
                    .then(pl.col("Bowler_Balls"))
                    .otherwise(None)
                )
                .fill_null(0)
                .mul(100)
                .alias("Dot_Ball_Percentage"),
                pl.when(pl.col("Bowler_Wkts") > 0)
                .then((pl.col("Bowler_Balls") / pl.col("Bowler_Wkts")).round(2))
                .otherwise(pl.col("Bowler_Balls"))
                .alias("Strike_Rate"),
                pl.when(pl.col("Bowler_Wkts") > 0)
                .then((pl.col("Bowler_Runs") / pl.col("Bowler_Wkts")).round(2))
                .otherwise(pl.col("Bowler_Runs"))
                .alias("Average"),
            ]
        )

        merged_pl = merged_pl.drop(["Bowler_Overs_Orig", "Bowler_Econ_Orig"])

        bowl_df = merged_pl.to_pandas()
        bowl_df = bowl_df[bowl_df["Name"].astype(str).str.strip() != ""]

        print(
            f"🔗 Bowling merge and calculations completed in {time.perf_counter() - merge_start:.2f} seconds"
        )

    _coalesce_column(bowl_df, "Bat_Team")
    _coalesce_column(bowl_df, "Bowl_Team")

    for column in ["Bat_Team", "Bowl_Team", "Home_Team", "Away_Team", "Player_of_the_Match"]:
        if column in bowl_df.columns:
            bowl_df[column] = bowl_df[column].fillna("").astype(str)

    if "Year" in bowl_df.columns:
        bowl_df["Year"] = bowl_df["Year"].fillna("0").astype(str)

    total_time = time.perf_counter() - start_time
    print(
        f"✅ Fast bowling processing completed in {total_time:.2f} seconds"
        f" (records={len(bowl_df)}, files={len(file_cache)})"
    )

    return bowl_df


def benchmark_processing_speed(directory_path: str, use_fast: bool = True):
    """Compare legacy processors against the fast pipeline."""

    match_df = process_match_data(directory_path)
    game_df = process_game_stats(directory_path, match_df)

    if use_fast:
        print("🚀 Benchmarking FAST processing...")
        start = time.perf_counter()
        processor = FastCricketProcessor()
        _, bowl_df, bat_df = processor.process_all_scorecards(directory_path)
        elapsed = time.perf_counter() - start
        print(f"⚡ FAST processing: {elapsed:.2f} seconds")
        return bat_df, bowl_df, elapsed

    from bat import process_bat_stats
    from bowl import process_bowl_stats

    print("🐌 Benchmarking ORIGINAL processing...")
    start = time.perf_counter()
    bowl_legacy = process_bowl_stats(directory_path, game_df, match_df)
    bat_legacy = process_bat_stats(directory_path, game_df, match_df)
    elapsed = time.perf_counter() - start
    print(f"🕐 ORIGINAL processing: {elapsed:.2f} seconds")
    return bat_legacy, bowl_legacy, elapsed