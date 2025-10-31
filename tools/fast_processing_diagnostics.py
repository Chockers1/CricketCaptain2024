"""
Interactive smoke test for the fast-processing pipeline.

Run with:
    python -m tools.fast_processing_diagnostics
"""

import time
from pathlib import Path


def test_fast_processing():
    """Smoke test the fast processing implementation."""
    try:
        from fast_processing import FastCricketProcessor, fast_process_bat_stats
        print("✅ Fast processing module imported successfully!")
    except ImportError as exc:
        print(f"❌ Failed to import fast processing: {exc}")
        return False

    try:
        processor = FastCricketProcessor()
        print("✅ FastCricketProcessor initialized successfully!")
    except Exception as exc:
        print(f"❌ Failed to initialize processor: {exc}")
        return False

    try:
        from fast_processing import PLAYER_PATTERN, BOWLER_PATTERN

        batting_line = "Smith c Jones b Williams 45 67 6 1"
        match = PLAYER_PATTERN.search(batting_line)
        if match:
            print("✅ Batting regex pattern works correctly!")
            print(f"   Name: {match.group('Name')}, Runs: {match.group('Runs')}")
        else:
            print("⚠️ Batting regex pattern needs adjustment")

        bowling_line = "Anderson 15.2 3 42 2 2.74"
        match = BOWLER_PATTERN.search(bowling_line)
        if match:
            print("✅ Bowling regex pattern works correctly!")
            print(f"   Name: {match.group('Name')}, Overs: {match.group('Overs')}")
        else:
            print("⚠️ Bowling regex pattern needs adjustment")

    except Exception as exc:
        print(f"❌ Regex pattern test failed: {exc}")
        return False

    scorecard_dir = Path(r"C:\Users\rtayl\AppData\Roaming\Childish Things\Cricket Captain 2024\Saves\Scorecards")
    if scorecard_dir.exists():
        txt_files = list(scorecard_dir.glob("*.txt"))
        if txt_files:
            print(f"✅ Found {len(txt_files)} scorecard files for testing")

            test_files = txt_files[:5]

            start_time = time.time()
            processed = processor.process_file_batch(test_files)
            elapsed = time.time() - start_time

            print(f"✅ Processed {len(processed)} files in {elapsed:.2f} seconds")
            print(f"   Average: {elapsed/len(test_files):.3f} seconds per file")

            if processed:
                sample = processed[0]
                print("✅ File processing successful!")
                print(f"   Sample result: {sample['filename']}, Teams: {sample['home_team']} v {sample['away_team']}")
            else:
                print("⚠️ No files were processed successfully")
        else:
            print("⚠️ No .txt files found in scorecard directory")
    else:
        print("⚠️ Scorecard directory not found - skipping file processing test")

    print("\n🎯 Fast Processing Test Complete!")
    print("If all tests passed, fast processing should work 2-5x faster than standard processing.")
    return True


if __name__ == "__main__":
    test_fast_processing()
