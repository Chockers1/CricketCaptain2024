"""
Cricket Captain Performance Utils
Utility functions for optimizing performance across the application
"""

import pandas as pd
import polars as pl
import streamlit as st
import gc
import sys
import time
from functools import wraps
from typing import Optional, Dict, Any, List, Callable
import hashlib
import json

class PerformanceManager:
    """Central performance management for Cricket Captain app"""
    
    def __init__(self):
        self.memory_limit = 200_000_000  # 200MB limit for session state
        self.chunk_size = 10_000  # Default chunk size for processing
        self.metadata_key = '_perf_session_meta'
        self.metrics_key = '_perf_session_metrics'
        self.cleanup_interval_seconds = 45
        self.access_grace_seconds = 60
        self.protected_keys = {'data_loaded', 'use_fast_processing', 'fast_processing_error'}
        self.cache_manager: Optional["CacheManager"] = None

    # ------------------------------------------------------------------
    # DataFrame registration & optimization utilities
    # ------------------------------------------------------------------
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting numeric types"""
        if df.empty:
            return df

        optimized_df = df.copy()

        for col in optimized_df.select_dtypes(include=['int64', 'Int64']):
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')

        for col in optimized_df.select_dtypes(include=['float64', 'Float64']):
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')

        for col in optimized_df.select_dtypes(include=['object']):
            if len(optimized_df) and optimized_df[col].nunique(dropna=False) / len(optimized_df) < 0.1:
                optimized_df[col] = optimized_df[col].astype('category')

        return optimized_df

    def register_dataframe(
        self,
        key: str,
        df: pd.DataFrame,
        *,
        optimize: bool = True,
        invalidate_cache: bool = True,
        reason: Optional[str] = None,
    ) -> pd.DataFrame:
        """Store a DataFrame in session state with metadata tracking."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("register_dataframe expects a pandas DataFrame")

        stored_df = self.optimize_dataframe(df) if optimize else df
        st.session_state[key] = stored_df

        metadata = self._get_metadata_store()
        now = time.time()
        size_bytes = stored_df.memory_usage(deep=True).sum()

        stored_df.attrs['_perf_session_key'] = key
        stored_df.attrs['_perf_last_access'] = now

        metadata[key] = {
            'size_bytes': size_bytes,
            'rows': int(len(stored_df)),
            'columns': int(len(stored_df.columns)),
            'last_updated': now,
            'last_access': now,
            'access_count': 0,
            'hash': self.get_data_hash(stored_df),
            'reason': reason or 'register_dataframe',
        }

        self._record_usage_snapshot()

        if invalidate_cache and self.cache_manager:
            data_type = self._data_type_from_key(key)
            if data_type:
                self.cache_manager.invalidate_related_caches(data_type, reason or 'data_update')

        return stored_df

    def touch_dataframe(self, key: str):
        """Mark a session DataFrame as accessed for LRU accounting."""
        metadata = self._get_metadata_store()
        if key not in metadata:
            if key in st.session_state and isinstance(st.session_state[key], pd.DataFrame):
                self.register_dataframe(key, st.session_state[key], optimize=False, invalidate_cache=False, reason='touch_auto_register')
            return

        now = time.time()
        metadata[key]['last_access'] = now
        metadata[key]['access_count'] = metadata[key].get('access_count', 0) + 1

        df = st.session_state.get(key)
        if isinstance(df, pd.DataFrame):
            df.attrs['_perf_last_access'] = now

    def remove_dataframe(self, key: str):
        """Remove a tracked DataFrame from session state and metadata."""
        metadata = self._get_metadata_store()
        if key in st.session_state:
            del st.session_state[key]
        metadata.pop(key, None)

    # ------------------------------------------------------------------
    # Session metadata helpers
    # ------------------------------------------------------------------
    def _get_metadata_store(self) -> Dict[str, Dict[str, Any]]:
        if self.metadata_key not in st.session_state:
            st.session_state[self.metadata_key] = {}
        return st.session_state[self.metadata_key]

    def _get_metrics_store(self) -> Dict[str, Any]:
        if self.metrics_key not in st.session_state:
            st.session_state[self.metrics_key] = {
                'history': [],
                'evictions': [],
                'last_cleanup': 0.0,
            }
        return st.session_state[self.metrics_key]

    def _record_usage_snapshot(self, total_override: Optional[int] = None):
        metadata = self._get_metadata_store()
        metrics = self._get_metrics_store()
        total_bytes = total_override if total_override is not None else sum(
            entry.get('size_bytes', 0) for entry in metadata.values()
        )
        metrics['history'].append({'timestamp': time.time(), 'total_bytes': total_bytes})
        metrics['history'] = metrics['history'][-50:]

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        metrics = self._get_metrics_store().copy()
        metrics['history'] = list(metrics.get('history', []))
        metrics['evictions'] = list(metrics.get('evictions', []))[-20:]
        return metrics

    # ------------------------------------------------------------------
    # Memory usage & cleanup
    # ------------------------------------------------------------------
    def get_memory_usage(self, recalc: bool = False) -> Dict[str, int]:
        """Get current memory usage of session state DataFrames."""
        usage: Dict[str, int] = {}
        total_size = 0
        metadata = self._get_metadata_store()

        for key, value in list(st.session_state.items()):
            if not isinstance(value, pd.DataFrame):
                continue

            if key not in metadata or recalc:
                size = value.memory_usage(deep=True).sum()
                metadata[key] = {
                    **metadata.get(key, {}),
                    'size_bytes': size,
                    'rows': int(len(value)),
                    'columns': int(len(value.columns)),
                    'last_updated': time.time(),
                    'last_access': metadata.get(key, {}).get('last_access', time.time()),
                    'hash': self.get_data_hash(value),
                }
            else:
                size = metadata[key].get('size_bytes', value.memory_usage(deep=True).sum())

            usage[key] = size
            total_size += size

        usage['total'] = total_size
        self._record_usage_snapshot(total_size)
        return usage

    def cleanup_session_state(self, force: bool = False) -> bool:
        """Clean up session state if memory usage exceeds limit."""
        usage = self.get_memory_usage(recalc=True)
        total = usage.get('total', 0)

        if not force and total <= self.memory_limit:
            return False

        metadata = self._get_metadata_store()
        metrics = self._get_metrics_store()
        now = time.time()

        candidates = []
        for key, value in list(st.session_state.items()):
            if not isinstance(value, pd.DataFrame):
                continue
            if key in self.protected_keys or key.startswith('_'):
                continue
            meta = metadata.get(key, {})
            last_access = meta.get('last_access', meta.get('last_updated', 0))
            if now - last_access < self.access_grace_seconds and not force:
                continue
            size = meta.get('size_bytes', value.memory_usage(deep=True).sum())
            candidates.append((last_access, size, key))

        candidates.sort(key=lambda item: (item[0], -item[1]))

        freed_bytes = 0
        evicted = []
        for last_access, size, key in candidates:
            if total - freed_bytes <= self.memory_limit and not force:
                break

            self.remove_dataframe(key)
            freed_bytes += size
            evicted.append({'key': key, 'size_bytes': size, 'last_access': last_access, 'timestamp': now})

        if evicted:
            metrics['evictions'].extend(evicted)
            metrics['evictions'] = metrics['evictions'][-50:]
            metrics['last_cleanup'] = now
            self._record_usage_snapshot(total - freed_bytes)
        else:
            metrics['last_cleanup'] = now

        gc.collect()
        return bool(evicted)

    def maybe_cleanup(self):
        """Run cleanup on a timed interval to act as a background tick."""
        metrics = self._get_metrics_store()
        now = time.time()
        if now - metrics.get('last_cleanup', 0) >= self.cleanup_interval_seconds:
            self.cleanup_session_state()
    
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
                hash_data['head'] = json.loads(
                    df.head(5).to_json(orient='split', date_format='iso')
                )
                hash_data['tail'] = json.loads(
                    df.tail(5).to_json(orient='split', date_format='iso')
                )
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _data_type_from_key(self, key: str) -> Optional[str]:
        mapping = {
            'bat_df': 'batting',
            'bowl_df': 'bowling',
            'match_df': 'match',
            'game_df': 'game',
        }
        return mapping.get(key)

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

    def __init__(self, performance_manager: Optional[PerformanceManager] = None):
        self.performance_manager = performance_manager
        self.registry_key = '_perf_cache_registry'
        self.stats_key = '_perf_cache_stats'
        self._runtime_registry: Dict[str, Callable[[], None]] = {}

    def _get_registry(self) -> Dict[str, Any]:
        if self.registry_key not in st.session_state:
            st.session_state[self.registry_key] = {
                'by_name': {},
                'by_type': {},
            }
        return st.session_state[self.registry_key]

    def _get_stats(self) -> Dict[str, Any]:
        if self.stats_key not in st.session_state:
            st.session_state[self.stats_key] = {}
        return st.session_state[self.stats_key]

    def register_cache(self, name: str, data_types: Optional[List[str]], clear_func: Callable[[], None]):
        registry = self._get_registry()
        by_name = registry['by_name']
        by_type = registry['by_type']

        by_name[name] = {
            'data_types': data_types or ['__global__'],
            'created': time.time(),
        }
        self._runtime_registry[name] = clear_func

        for dtype in by_name[name]['data_types']:
            bucket = by_type.setdefault(dtype, [])
            if name not in bucket:
                bucket.append(name)

    def record_request(self, name: str, signature: str):
        stats = self._get_stats().setdefault(name, {'hits': 0, 'requests': 0, 'last_signature': None})
        stats['requests'] += 1
        stats['last_signature'] = signature

    def record_hit(self, name: str, signature: str):
        stats = self._get_stats().setdefault(name, {'hits': 0, 'requests': 0, 'last_signature': None})
        stats['hits'] += 1
        stats['last_signature'] = signature

    def record_materialized(self, name: str, signature: str, result: Any):
        stats = self._get_stats().setdefault(name, {
            'hits': 0,
            'requests': 0,
            'last_signature': None,
            'last_size_bytes': 0,
        })
        size_bytes = 0
        if isinstance(result, pd.DataFrame):
            size_bytes = result.memory_usage(deep=True).sum()
        stats['last_size_bytes'] = size_bytes
        stats['last_signature'] = signature

    def invalidate_related_caches(self, data_type: str, reason: Optional[str] = None):
        registry = self._get_registry()
        by_name = registry.get('by_name', {})
        by_type = registry.get('by_type', {})
        names = list(by_type.get(data_type, []))

        for name in names:
            entry = by_name.get(name)
            if not entry:
                continue
            clear_func = self._runtime_registry.get(name)
            if clear_func:
                clear_func()
                entry['last_invalidated'] = time.time()
                entry['reason'] = reason or 'manual'

    def describe(self) -> Dict[str, Any]:
        stats = {}
        for name, data in self._get_stats().items():
            stats[name] = {
                **data,
                'misses': max(data.get('requests', 0) - data.get('hits', 0), 0),
            }
        return {
            'registry': self._get_registry(),
            'stats': stats,
        }

# Global performance manager instance
perf_manager = PerformanceManager()
cache_manager = CacheManager(performance_manager=perf_manager)
perf_manager.cache_manager = cache_manager
cache_manager.performance_manager = perf_manager


# Utility decorators for easy use
def optimize_dataframe_decorator(func):
    """Decorator to automatically optimize DataFrames returned by functions"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            return perf_manager.optimize_dataframe(result)
        return result

    return wrapper


def memory_efficient_cache(ttl=None, data_types: Optional[List[str]] = None):
    """Memory-efficient caching decorator with data-hash invalidation."""

    def decorator(func):
        cache_name = func.__name__

        @st.cache_data(ttl=ttl)
        def _cached_call(_data_signature: str, *args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                result = perf_manager.optimize_dataframe(result)
            if cache_manager:
                cache_manager.record_materialized(cache_name, _data_signature, result)
            return result

        if cache_manager:
            cache_manager.register_cache(cache_name, data_types, _cached_call.clear)

        @wraps(func)
        def wrapper(*args, **kwargs):
            perf_manager.maybe_cleanup()
            data_signature = perf_manager.get_data_hash()
            if cache_manager:
                cache_manager.record_request(cache_name, data_signature)
            result = _cached_call(data_signature, *args, **kwargs)
            if cache_manager:
                cache_manager.record_hit(cache_name, data_signature)
            return result

        wrapper.clear = _cached_call.clear  # type: ignore[attr-defined]
        return wrapper

    return decorator


def get_tracked_dataframe(key: str) -> pd.DataFrame:
    """Convenience accessor that updates LRU metadata when reading session DataFrames."""
    value = st.session_state.get(key)
    if isinstance(value, pd.DataFrame):
        perf_manager.touch_dataframe(key)
        return value
    return pd.DataFrame()