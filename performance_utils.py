"""
Cricket Captain Performance Utils
Utility functions for optimizing performance across the application
"""

import pandas as pd
import polars as pl
import streamlit as st
import gc
import sys
from typing import Optional, Dict, Any, List
import hashlib
import json

class PerformanceManager:
    """Central performance management for Cricket Captain app"""
    
    def __init__(self):
        self.memory_limit = 200_000_000  # 200MB limit for session state
        self.chunk_size = 10_000  # Default chunk size for processing
        
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting numeric types"""
        if df.empty:
            return df
            
        # Make a copy to avoid modifying original
        optimized_df = df.copy()
        
        # Optimize integer columns
        for col in optimized_df.select_dtypes(include=['int64']):
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
            
        # Optimize float columns  
        for col in optimized_df.select_dtypes(include=['float64']):
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
            
        # Convert object columns to category if they have low cardinality
        for col in optimized_df.select_dtypes(include=['object']):
            if optimized_df[col].nunique() / len(optimized_df) < 0.1:  # Less than 10% unique values
                optimized_df[col] = optimized_df[col].astype('category')
                
        return optimized_df
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage of session state DataFrames"""
        usage = {}
        total_size = 0
        
        for key, value in st.session_state.items():
            if isinstance(value, pd.DataFrame):
                size = value.memory_usage(deep=True).sum()
                usage[key] = size
                total_size += size
                
        usage['total'] = total_size
        return usage
    
    def cleanup_session_state(self):
        """Clean up session state if memory usage exceeds limit"""
        usage = self.get_memory_usage()
        
        if usage['total'] > self.memory_limit:
            # Sort DataFrames by size (largest first)
            df_sizes = [(k, v) for k, v in usage.items() if k != 'total']
            df_sizes.sort(key=lambda x: x[1], reverse=True)
            
            # Remove largest DataFrames until under limit
            for key, size in df_sizes:
                if usage['total'] > self.memory_limit:
                    if key in st.session_state:
                        del st.session_state[key]
                        usage['total'] -= size
                        st.warning(f"Removed {key} from session state to free memory")
                        
        # Force garbage collection
        gc.collect()
    
    def get_data_hash(self, df: Optional[pd.DataFrame] = None) -> str:
        """Generate hash for caching based on DataFrame content"""
        if df is not None:
            # Hash based on shape and a sample of data
            hash_data = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': str(df.dtypes.to_dict()),
            }
            # Add sample of first/last rows for content verification
            if not df.empty:
                hash_data['head'] = df.head(5).to_dict()
                hash_data['tail'] = df.tail(5).to_dict()
        else:
            # Hash all DataFrames in session state
            hash_components = []
            for key in ['bat_df', 'bowl_df', 'match_df', 'game_df']:
                df_data = st.session_state.get(key)
                if isinstance(df_data, pd.DataFrame) and not df_data.empty:
                    hash_components.append(f"{key}:{df_data.shape}")
                    
            hash_data = {'components': hash_components}
            
        return hashlib.md5(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()
    
    def process_in_chunks(self, df: pd.DataFrame, process_func, chunk_size: Optional[int] = None) -> List:
        """Process DataFrame in chunks to manage memory"""
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        results = []
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        
        progress_bar = st.progress(0)
        for i, start in enumerate(range(0, len(df), chunk_size)):
            end = min(start + chunk_size, len(df))
            chunk = df.iloc[start:end]
            
            # Process chunk
            result = process_func(chunk)
            results.append(result)
            
            # Update progress
            progress_bar.progress((i + 1) / total_chunks)
            
            # Cleanup after each chunk
            gc.collect()
            
        progress_bar.empty()
        return results

class PolarsOptimizer:
    """Utilities for converting operations to Polars for better performance"""
    
    @staticmethod
    def pandas_to_polars_safe(df: pd.DataFrame) -> pl.DataFrame:
        """Safely convert pandas DataFrame to Polars"""
        try:
            # Handle problematic dtypes
            df_copy = df.copy()
            
            # Convert datetime columns
            for col in df_copy.select_dtypes(include=['datetime64']):
                df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d')
                
            # Convert category columns to string
            for col in df_copy.select_dtypes(include=['category']):
                df_copy[col] = df_copy[col].astype('str')
                
            return pl.from_pandas(df_copy)
        except Exception as e:
            st.error(f"Error converting to Polars: {e}")
            return pl.DataFrame()
    
    @staticmethod
    @st.cache_data
    def compute_batting_stats_polars(df: pd.DataFrame) -> pd.DataFrame:
        """Compute batting statistics using Polars for speed"""
        if df.empty:
            return pd.DataFrame()
            
        try:
            pl_df = PolarsOptimizer.pandas_to_polars_safe(df)
            
            # Add milestone flags using Polars
            pl_df = pl_df.with_columns([
                (pl.col('Runs') >= 50).cast(pl.Int64).alias('50s'),
                (pl.col('Runs') >= 100).cast(pl.Int64).alias('100s'),
                (pl.col('Runs') >= 200).cast(pl.Int64).alias('200s'),
            ])
            
            # Compute aggregated statistics
            result = (
                pl_df.group_by(['Name', 'Bat_Team_y'])
                .agg([
                    pl.col('File Name').n_unique().alias('Matches'),
                    pl.col('Batted').sum().alias('Innings'),
                    pl.col('Out').sum().alias('Outs'),
                    pl.col('Runs').sum().alias('Total_Runs'),
                    pl.col('Balls').sum().alias('Total_Balls'),
                    pl.col('Runs').max().alias('Highest_Score'),
                    pl.col('4s').sum().alias('Total_4s'),
                    pl.col('6s').sum().alias('Total_6s'),
                    pl.col('50s').sum().alias('Fifties'),
                    pl.col('100s').sum().alias('Hundreds'),
                    pl.col('200s').sum().alias('Double_Hundreds'),
                ])
                .with_columns([
                    # Calculate averages and rates
                    (pl.col('Total_Runs') / pl.col('Outs').clip(1)).round(2).alias('Average'),
                    (pl.col('Total_Runs') * 100 / pl.col('Total_Balls').clip(1)).round(2).alias('Strike_Rate'),
                    (pl.col('Total_Balls') / pl.col('Outs').clip(1)).round(2).alias('Balls_per_Dismissal'),
                ])
                .sort(['Total_Runs'], descending=True)
            )
            
            return result.to_pandas()
            
        except Exception as e:
            st.error(f"Error in Polars computation: {e}")
            return pd.DataFrame()

class CacheManager:
    """Intelligent cache management for Streamlit"""
    
    def __init__(self):
        self.cache_stats = {}
    
    def get_cache_key(self, function_name: str, *args, **kwargs) -> str:
        """Generate intelligent cache key"""
        # Include function name and data hash in key
        key_data = {
            'function': function_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items())),
            'data_hash': PerformanceManager().get_data_hash()
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def invalidate_related_caches(self, data_type: str):
        """Invalidate only caches related to specific data type"""
        if data_type == 'batting':
            # Clear only batting-related caches
            cache_keys_to_clear = [k for k in st.session_state.keys() 
                                 if 'bat' in k.lower() and 'cache' in k.lower()]
        elif data_type == 'bowling':
            cache_keys_to_clear = [k for k in st.session_state.keys() 
                                 if 'bowl' in k.lower() and 'cache' in k.lower()]
        else:
            return
            
        for key in cache_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

# Global performance manager instance
perf_manager = PerformanceManager()

# Utility decorators for easy use
def optimize_dataframe_decorator(func):
    """Decorator to automatically optimize DataFrames returned by functions"""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            return perf_manager.optimize_dataframe(result)
        return result
    return wrapper

def memory_efficient_cache(ttl=None):
    """Memory-efficient caching decorator that includes cleanup"""
    def decorator(func):
        @st.cache_data(ttl=ttl)
        def wrapper(*args, **kwargs):
            # Check memory before expensive operations
            perf_manager.cleanup_session_state()
            
            result = func(*args, **kwargs)
            
            # Optimize result if it's a DataFrame
            if isinstance(result, pd.DataFrame):
                result = perf_manager.optimize_dataframe(result)
                
            return result
        return wrapper
    return decorator