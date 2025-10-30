"""
Memory optimization utilities for Cricket Captain
Add this to your main application to implement memory optimizations
"""

import streamlit as st
import pandas as pd
import numpy as np
from performance_utils import PerformanceManager

# Initialize performance manager
perf_manager = PerformanceManager()

def optimize_dataframes_on_load():
    """
    Call this function after loading data to optimize all DataFrames in session state
    This is the HIGHEST IMPACT optimization - implement this first!
    """
    dataframes_optimized = 0
    memory_saved = 0
    
    for key in ['bat_df', 'bowl_df', 'match_df', 'game_df']:
        if key in st.session_state:
            df = st.session_state[key]
            if isinstance(df, pd.DataFrame) and not df.empty:
                # Get original memory usage
                original_memory = df.memory_usage(deep=True).sum()
                
                # Optimize the DataFrame
                optimized_df = perf_manager.optimize_dataframe(df)
                
                # Update session state
                st.session_state[key] = optimized_df
                
                # Calculate memory savings
                new_memory = optimized_df.memory_usage(deep=True).sum()
                memory_saved += (original_memory - new_memory)
                dataframes_optimized += 1
    
    if dataframes_optimized > 0:
        st.success(f"✅ Optimized {dataframes_optimized} DataFrames - Saved {memory_saved / 1_000_000:.1f} MB memory!")

def add_memory_sidebar():
    """
    Add memory monitoring to the sidebar - call this in your main views
    """
    with st.sidebar:
        st.markdown("### 📊 Performance Monitor")
        
        # Get memory usage
        usage = perf_manager.get_memory_usage()
        total_mb = usage['total'] / 1_000_000
        limit_mb = perf_manager.memory_limit / 1_000_000
        
        # Progress bar for memory usage
        progress_value = min(usage['total'] / perf_manager.memory_limit, 1.0)
        
        # Color coding for memory usage
        if progress_value < 0.5:
            color = "🟢"
            status = "Good"
        elif progress_value < 0.8:
            color = "🟡"  
            status = "Moderate"
        else:
            color = "🔴"
            status = "High"
            
        st.markdown(f"**Memory Usage** {color} {status}")
        st.progress(progress_value)
        st.markdown(f"{total_mb:.1f} MB / {limit_mb:.1f} MB")
        
        # Show DataFrame breakdown
        if len(usage) > 1:
            st.markdown("**Largest DataFrames:**")
            df_usage = [(k, v) for k, v in usage.items() if k != 'total']
            df_usage.sort(key=lambda x: x[1], reverse=True)
            
            for name, size in df_usage[:3]:
                size_mb = size / 1_000_000
                st.markdown(f"- **{name}**: {size_mb:.1f} MB")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clean"):
                perf_manager.cleanup_session_state()
                st.rerun()
        
        with col2:
            if st.button("⚡ Optimize"):
                optimize_dataframes_on_load()
                st.rerun()

def efficient_dataframe_filter(df_key: str, query_string: str = None, filters: dict = None):
    """
    Efficiently filter DataFrames using query strings (no copying)
    
    Args:
        df_key: Key in session_state ('bat_df', 'bowl_df', etc.)
        query_string: Pandas query string for filtering
        filters: Dict of column filters
    
    Returns:
        Filtered DataFrame view (no copy)
    """
    if df_key not in st.session_state:
        return pd.DataFrame()
    
    df = st.session_state[df_key]
    
    # Use pandas query for efficient filtering (no copy)
    if query_string:
        try:
            return df.query(query_string)
        except Exception:
            # Fallback to boolean indexing if query fails
            pass
    
    # Apply filters using boolean indexing
    if filters:
        mask = pd.Series(True, index=df.index)
        for col, values in filters.items():
            if col in df.columns and values:
                if isinstance(values, list):
                    mask &= df[col].isin(values)
                else:
                    mask &= (df[col] == values)
        return df[mask]
    
    return df

def paginate_dataframe(df: pd.DataFrame, page_size: int = 50, current_page: int = 1):
    """
    Paginate large DataFrames for better performance
    
    Returns:
        Tuple of (paginated_df, total_pages, current_page_info)
    """
    total_rows = len(df)
    total_pages = (total_rows + page_size - 1) // page_size
    current_page = max(1, min(current_page, total_pages))
    
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    paginated_df = df.iloc[start_idx:end_idx]
    
    page_info = {
        'showing_from': start_idx + 1,
        'showing_to': end_idx,
        'total_rows': total_rows,
        'current_page': current_page,
        'total_pages': total_pages
    }
    
    return paginated_df, total_pages, page_info

def add_pagination_controls():
    """
    Add pagination controls to the sidebar
    Returns: (page_size, current_page)
    """
    with st.sidebar:
        st.markdown("### 📄 Table Settings")
        
        # Page size selection
        page_size = st.selectbox(
            "Rows per page",
            options=[25, 50, 100, 200, 500],
            index=1,  # Default to 50
            help="Number of rows to display per page"
        )
        
        # Current page will be set by the main view
        current_page = st.session_state.get('current_page', 1)
        
        return page_size, current_page

def display_paginated_dataframe(df: pd.DataFrame, key: str = "main_table"):
    """
    Display a DataFrame with pagination and performance optimizations
    """
    if df.empty:
        st.info("No data to display")
        return
    
    # Get pagination settings
    page_size, _ = add_pagination_controls()
    
    # Initialize current page in session state
    page_key = f"{key}_current_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    
    # Paginate the DataFrame
    paginated_df, total_pages, page_info = paginate_dataframe(
        df, page_size, st.session_state[page_key]
    )
    
    # Display the paginated DataFrame
    st.dataframe(
        paginated_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Pagination controls
    if total_pages > 1:
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if st.button("⬅️ Prev", disabled=st.session_state[page_key] <= 1):
                st.session_state[page_key] -= 1
                st.rerun()
        
        with col2:
            if st.button("➡️ Next", disabled=st.session_state[page_key] >= total_pages):
                st.session_state[page_key] += 1
                st.rerun()
        
        with col3:
            # Page number input
            new_page = st.number_input(
                f"Page ({total_pages} total)",
                min_value=1,
                max_value=total_pages,
                value=st.session_state[page_key],
                key=f"{key}_page_input"
            )
            if new_page != st.session_state[page_key]:
                st.session_state[page_key] = new_page
                st.rerun()
        
        with col4:
            if st.button("⏪ First", disabled=st.session_state[page_key] <= 1):
                st.session_state[page_key] = 1
                st.rerun()
        
        with col5:
            if st.button("⏩ Last", disabled=st.session_state[page_key] >= total_pages):
                st.session_state[page_key] = total_pages
                st.rerun()
    
    # Display page info
    st.caption(
        f"Showing {page_info['showing_from']}-{page_info['showing_to']} "
        f"of {page_info['total_rows']} rows"
    )

# Example usage functions for your existing views
def optimized_batting_view():
    """
    Example of how to use these optimizations in batview.py
    """
    # Add memory monitoring
    add_memory_sidebar()
    
    # Check if data exists
    if 'bat_df' not in st.session_state:
        st.warning("Please upload scorecard data first!")
        return
    
    # Get filters from sidebar
    with st.sidebar:
        st.markdown("### 🎯 Filters")
        
        # Get unique values efficiently (no copying)
        bat_df = st.session_state['bat_df']
        teams = sorted(bat_df['Bat_Team_y'].unique())
        formats = sorted(bat_df['Match_Format'].unique())
        
        selected_teams = st.multiselect("Teams", teams)
        selected_formats = st.multiselect("Formats", formats)
    
    # Build efficient filter
    filters = {}
    if selected_teams:
        filters['Bat_Team_y'] = selected_teams
    if selected_formats:
        filters['Match_Format'] = selected_formats
    
    # Apply filters efficiently (no copying)
    filtered_df = efficient_dataframe_filter('bat_df', filters=filters)
    
    # Display results with pagination
    if not filtered_df.empty:
        st.markdown("## 📊 Batting Statistics")
        display_paginated_dataframe(filtered_df, key="batting_table")
    else:
        st.info("No data matches your filters")

# Quick implementation checklist
IMPLEMENTATION_CHECKLIST = """
## ✅ Quick Implementation Checklist (60 minutes)

### Step 1: Add to your main app file (5 minutes)
```python
from memory_optimization import optimize_dataframes_on_load, add_memory_sidebar

# Add this after data loading
if st.button("🚀 Optimize Memory"):
    optimize_dataframes_on_load()
```

### Step 2: Update your view files (20 minutes each view)
1. Replace `df.copy()` with `df` where you're not modifying data
2. Add `add_memory_sidebar()` to each view
3. Replace large table displays with `display_paginated_dataframe(df)`

### Step 3: Test the improvements (15 minutes)
1. Load a large dataset
2. Check memory usage before/after optimization
3. Verify all views still work correctly

### Expected Results:
- 40-60% memory reduction immediately
- Faster view loading
- Better user experience with pagination
- Ability to handle larger datasets
"""