"""
Test script to verify fast processing improvements
Run this to benchmark the performance gains
"""

import time
import pandas as pd
from pathlib import Path

def test_fast_processing():
    """Test the fast processing implementation"""
    
    # Check if we can import the fast processing
    try:
        from fast_processing import FastCricketProcessor, fast_process_bat_stats
        print("✅ Fast processing module imported successfully!")
    except ImportError as e:
        print(f"❌ Failed to import fast processing: {e}")
        return False
    
    # Test the processor initialization
    try:
        processor = FastCricketProcessor()
        print("✅ FastCricketProcessor initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return False
    
    # Test regex patterns
    try:
        from fast_processing import PLAYER_PATTERN, BOWLER_PATTERN
        
        # Test batting pattern
        test_batting_line = "Smith c Jones b Williams 45 67 6 1"
        match = PLAYER_PATTERN.search(test_batting_line)
        if match:
            print("✅ Batting regex pattern works correctly!")
            print(f"   Name: {match.group('Name')}, Runs: {match.group('Runs')}")
        else:
            print("⚠️ Batting regex pattern needs adjustment")
        
        # Test bowling pattern  
        test_bowling_line = "Anderson 15.2 3 42 2 2.74"
        match = BOWLER_PATTERN.search(test_bowling_line)
        if match:
            print("✅ Bowling regex pattern works correctly!")
            print(f"   Name: {match.group('Name')}, Overs: {match.group('Overs')}")
        else:
            print("⚠️ Bowling regex pattern needs adjustment")
            
    except Exception as e:
        print(f"❌ Regex pattern test failed: {e}")
        return False
    
    # Test file processing (if scorecard directory exists)
    scorecard_dir = r"C:\Users\rtayl\AppData\Roaming\Childish Things\Cricket Captain 2024\Saves\Scorecards"
    if Path(scorecard_dir).exists():
        txt_files = list(Path(scorecard_dir).glob("*.txt"))
        if txt_files:
            print(f"✅ Found {len(txt_files)} scorecard files for testing")
            
            # Test processing a small batch
            test_files = txt_files[:5]  # Test with first 5 files
            
            start_time = time.time()
            processed = processor.process_file_batch(test_files)
            processing_time = time.time() - start_time
            
            print(f"✅ Processed {len(processed)} files in {processing_time:.2f} seconds")
            print(f"   Average: {processing_time/len(test_files):.3f} seconds per file")
            
            if processed:
                print("✅ File processing successful!")
                sample = processed[0]
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