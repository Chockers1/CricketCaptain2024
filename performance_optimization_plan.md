# Cricket Captain Performance Optimization Plan

## 🎯 Priority 1: Memory Optimization (High Impact)

### 1.1 DataFrame Memory Management (DONE)
- **Issue**: Excessive copying of large DataFrames in session state
- **Solution**: Implement copy-on-write and view-based data access
- **Impact**: 40-60% memory reduction

```python
# Before (inefficient)
bat_df = st.session_state['bat_df'].copy()

# After (efficient)
@st.cache_data
def get_filtered_bat_data(filters_hash):
    return st.session_state['bat_df'].query(filters)  # No copy
```

### 1.2 Session State Optimization
- **Issue**: Large DataFrames stored indefinitely in session state
- **Solution**: Implement session state size limits and cleanup
- **Impact**: 50-70% memory reduction

```python
def cleanup_session_state():
    """Remove large unused DataFrames from session state"""
    size_limit = 100_000_000  # 100MB limit
    current_size = sum(df.memory_usage(deep=True).sum() 
                      for df in st.session_state.values() 
                      if isinstance(df, pd.DataFrame))
    if current_size > size_limit:
        # Remove least recently used DataFrames
        pass
```

### 1.3 Data Type Optimization
- **Issue**: Using default int64/float64 for all numeric columns
- **Solution**: Downcast to appropriate types
- **Impact**: 30-50% memory reduction

```python
def optimize_dtypes(df):
    """Optimize DataFrame dtypes for memory efficiency"""
    for col in df.select_dtypes(include=['int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df
```

## 🚀 Priority 2: Processing Speed (High Impact)

### 2.1 Polars Migration
- **Issue**: Mixed pandas/polars usage, pandas bottlenecks
- **Solution**: Migrate heavy computations to Polars
- **Impact**: 2-10x speed improvement
- **Progress (Oct 31 2025)**: Bowling and batting tabs now share a single Polars-backed cache via `compute_bowl_metrics` and the new `compute_bat_metrics` pipeline; remaining candidates are Team, Rankings, and Records secondary summaries.

```python
# Convert to consistent Polars usage
@st.cache_data
def compute_batting_stats_polars(df):
    pl_df = pl.from_pandas(df)
    return (
        pl_df.group_by(['Name', 'Team'])
        .agg([
            pl.col('Runs').sum().alias('Total_Runs'),
            pl.col('Balls').sum().alias('Total_Balls'),
            # More aggregations...
        ])
        .to_pandas()
    )
```

### 2.2 Fast Scorecard Ingestion (DONE)
- **Issue**: Legacy `process_*` scripts required several seconds for large archives and duplicated parsing work for each run.
- **Solution**: Introduce `FastCricketProcessor` + Polars-backed `fast_process_bat_stats` / `fast_process_bowl_stats` with cached scorecard parsing.
- **Impact**: 2-5× faster ingestion; 3,208 scorecards (88,825 batting rows / 40,797 bowling rows) now complete in <1.0s per discipline after the initial 0.4s text load.
- **Status update (Oct 31 2025)**: Streamlit now toggles `use_fast_processing` so Bat View logs capture cache hits and <0.1s refreshes on repeated interactions.

```python
from fast_processing import FastCricketProcessor

processor = FastCricketProcessor()
game_df, bowl_df, bat_df = processor.process_all_scorecards(scorecard_dir)
```

### 2.3 Chunk Processing
- **Issue**: Loading entire datasets into memory at once
- **Solution**: Process data in chunks
- **Impact**: Handles 10x larger datasets

```python
def process_large_dataset(df, chunk_size=10000):
    """Process large DataFrame in chunks"""
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        yield process_chunk(chunk)
```

### 2.4 Vectorized Operations (IN PROGRESS ✅ Records View hot paths)
- **Issue**: Loop-based calculations in views
- **Solution**: Replace with vectorized pandas/numpy operations
- **Status update (Oct 30 2025)**: Rare achievements in Records View (hundreds/5w in each innings) now use filtered pivots with `observed=True`, eliminating row-wise lambdas and drastically reducing load times.
- **Next**: Apply the same strategy to remaining streak calculators that still iterate row-by-row.
- **Impact**: 5-20x speed improvement

## 🔧 Priority 3: Caching Strategy (Medium Impact)

### 3.1 Intelligent Cache Management
- **Current**: Cache expiry set to 72 hours
- **Improved**: Data-dependent cache invalidation
- **Impact**: Better performance with current data

```python
@st.cache_data(ttl=None)  # Never expire based on time
def cached_computation(data_hash, *args):
    # Cache based on data hash, not time
    return expensive_computation(*args)

def get_data_hash():
    return hash(tuple(st.session_state.get('bat_df', pd.DataFrame()).shape))
```

### 3.2 Selective Cache Clearing
- **Issue**: Clearing entire cache when data changes
- **Solution**: Clear only affected caches
- **Impact**: Preserve unrelated cached computations

## 📊 Priority 4: UI Performance (Medium Impact)

### 4.1 Lazy Loading (DONE ✅)
- **Issue**: All views load data immediately
- **Solution**: Load data only when tabs are accessed
- **Status update (Oct 2025)**: Records tab now uses a “Load Records Data” gate in session state; heavy calculations defer until requested.
- **Impact**: Faster initial page load

### 4.2 Pagination for Large Tables
- **Issue**: Displaying thousands of rows at once
- **Solution**: Implement pagination or virtual scrolling
- **Impact**: Faster rendering, lower memory usage

## 🔍 Priority 5: Code Optimization (Low-Medium Impact)

### 5.1 Regex Compilation
- **Issue**: Regex compiled in loops
- **Solution**: Pre-compile regex patterns
- **Impact**: 2-5x faster text processing

```python
# Pre-compile patterns
PLAYER_PATTERN = re.compile(r"^(?P<Name>.+?)...")
DATE_PATTERNS = [re.compile(pattern) for pattern in date_patterns]
```

### 5.2 Database Integration
- **Issue**: All data in memory/session state
- **Solution**: SQLite for large datasets
- **Impact**: Handle unlimited data sizes

## 📈 Implementation Priority Matrix

| Enhancement | Impact | Effort | Priority | Status |
|-------------|--------|--------|----------|--------|
| DataFrame Memory Management | High | Medium | 1 | ✅ Complete |
| Polars Migration | High | High | 2 | 🔄 In progress (Bowling + Batting views live; PlayerRankings next, Records after) |
| Fast Scorecard Ingestion | High | Medium | 2a | ✅ Complete |
| Session State Cleanup | High | Low | 3 | ⏳ Planned |
| Chunk Processing | High | Medium | 4 | ⏳ Planned |
| Cache Strategy | Medium | Low | 5 | ⏳ Planned |
| Data Type Optimization | Medium | Low | 6 | ⏳ Planned |
| UI Lazy Loading | Medium | Medium | 7 | ✅ Records tab |
| Regex Optimization | Low | Low | 8 | ⏳ Not started |

## 🎯 Expected Performance Gains

### Memory Usage:
- **Current**: ~500MB for 1000 scorecards
- **Optimized**: ~150MB for same dataset (70% reduction)

### Processing Speed:
- **File Processing**: Fast ingest completes 3,200+ scorecards in ~0.9s after cached parsing (2-5× faster than legacy)
- **View Loading**: 3-5x faster with optimized caching
- **Filtering**: 5-10x faster with vectorized operations

### Scalability:
- **Current**: Handles ~2000 scorecards reliably
- **Optimized**: Handle 10,000+ scorecards without issues

## 📋 Next Steps

1. **Finish Polars migration** for Team, Rankings, and Records secondary summaries.
2. **Implement session-state cleanup** and chunk processing for mega archives.
3. **Tighten cache strategy** with data-hash invalidation.
4. **Run benchmark_performance.py** during releases to guard fast ingest regressions.
5. **Collect user feedback** on perceived responsiveness.

## ✅ Recent Updates (Oct 31 2025)

- Validated `FastCricketProcessor` across 3,208 scorecards with sub-second aggregation and integrated timing logs into Bat View cache hits.
- Streamlined Records View rare-achievement tables (double hundreds and double five-wicket hauls) with vectorized pivots and long-format filtering (Oct 30 2025).
- Added `observed=True` safeguards and day-first date parsing across Records View aggregations to eliminate MemoryErrors and warnings (Oct 30 2025).
- Enabled lazy loading control for the Records tab, preventing heavy calculations until requested (Oct 30 2025).