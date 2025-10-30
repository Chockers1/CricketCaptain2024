"""
Quick performance benchmark test for Fast Processing vs Standard Processing
"""

import time
import tempfile
import shutil
import os
from pathlib import Path

# Import both systems
from fast_processing import FastCricketProcessor
from game import process_game_stats
from bat import process_bat_stats
from bowl import process_bowl_stats

def time_standard_processing(temp_dir):
    """Time the standard processing approach"""
    print("🐌 Testing Standard Processing...")
    start = time.time()
    
    try:
        game_df = process_game_stats(temp_dir)
        print(f"   Standard Game Stats: {time.time() - start:.2f}s")
        
        bowl_start = time.time()
        bowl_df = process_bowl_stats(temp_dir)
        print(f"   Standard Bowl Stats: {time.time() - bowl_start:.2f}s")
        
        bat_start = time.time() 
        bat_df = process_bat_stats(temp_dir)
        print(f"   Standard Bat Stats: {time.time() - bat_start:.2f}s")
        
    except Exception as e:
        print(f"   Error in standard processing: {e}")
        return None
    
    total = time.time() - start
    print(f"   Standard Total: {total:.2f}s")
    return total

def time_fast_processing(temp_dir):
    """Time the fast processing approach"""
    print("🚀 Testing Fast Processing...")
    start = time.time()
    
    try:
        processor = FastCricketProcessor()
        game_df, bowl_df, bat_df = processor.process_all_scorecards(temp_dir)
        
    except Exception as e:
        print(f"   Error in fast processing: {e}")
        return None
    
    total = time.time() - start
    print(f"   Fast Total: {total:.2f}s")
    return total

def main():
    print("⚡ Cricket Captain Performance Benchmark")
    print("=" * 50)
    
    # Use the existing temp directory with scorecard files
    temp_dir = Path(tempfile.gettempdir()) / "cricket_scorecards"
    
    if not temp_dir.exists():
        print("❌ No test data found. Run the main processing first.")
        return
    
    # Count files
    txt_files = list(temp_dir.glob("*.txt"))
    print(f"📂 Testing with {len(txt_files)} scorecard files")
    print()
    
    # Test standard processing
    standard_time = time_standard_processing(str(temp_dir))
    print()
    
    # Test fast processing  
    fast_time = time_fast_processing(str(temp_dir))
    print()
    
    if standard_time and fast_time:
        speedup = standard_time / fast_time
        print("🏆 PERFORMANCE RESULTS:")
        print(f"   Standard Processing: {standard_time:.2f}s")
        print(f"   Fast Processing:     {fast_time:.2f}s")
        print(f"   Speed Improvement:   {speedup:.1f}x faster!")
        
        if speedup >= 2.0:
            print("   ✅ EXCELLENT: Achieved 2x+ speed improvement!")
        elif speedup >= 1.5:
            print("   ✅ GOOD: Significant speed improvement achieved!")
        else:
            print("   ⚠️  MODEST: Some improvement, but could be better")
    else:
        print("❌ Could not complete benchmark comparison")

if __name__ == "__main__":
    main()