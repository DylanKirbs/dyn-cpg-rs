import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import glob
import os
from pathlib import Path
import re
from typing import Dict, Any

# Configure page
st.set_page_config(page_title="CPG Benchmark Explorer", layout="wide", page_icon="🔍")

# ======================
# 1. DATA LOADING & PREP
# ======================

@st.cache_data
def load_data(directory: str):
    pattern = os.path.join(directory, "benchmark_*.json")
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame(), {}

    all_rows = []
    file_info = {}

    for file_path in files:
        filename = Path(file_path).name
        parts = filename[:-5].split('_')  # Remove .json
        repo = parts[1] if len(parts) > 1 else "unknown"
        tactic = '_'.join(parts[2:]) if len(parts) > 2 else "unknown"
        file_info[filename] = {"repo": repo, "tactic": tactic, "path": file_path}

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            for commit in data:
                step = commit['step']
                commit_id = commit['commit']
                for file_data in commit.get('files', []):
                    row = {
                        'step': step,
                        'commit': commit_id[:8],
                        'file': file_data.get('file', 'unknown'),
                        'repo': repo,
                        'tactic': tactic,
                        'unchanged': file_data.get('unchanged', False),
                        'full_nodes': file_data.get('full_nodes'),
                        'full_edges': file_data.get('full_edges'),
                        'incremental_nodes': file_data.get('incremental_nodes'),
                        'incremental_edges': file_data.get('incremental_edges'),
                        'comparison_result': file_data.get('comparison_result'),
                        'full_parse_time_ms': file_data.get('full_parse_time_ms'),
                        'incremental_parse_time_ms': file_data.get('incremental_parse_time_ms'),
                        'cpg_update_time_ms': file_data.get('cpg_update_time_ms'),
                        'comparison_time_ms': file_data.get('comparison_time_ms'),
                    }
                    # Extract detailed timings
                    if 'detailed_timings' in file_data and isinstance(file_data['detailed_timings'], dict):
                        for k, v in file_data['detailed_timings'].items():
                            row[f'timing_{k}'] = v
                    
                    # Extract file metrics
                    if 'file_metrics' in file_data and isinstance(file_data['file_metrics'], dict):
                        for k, v in file_data['file_metrics'].items():
                            row[f'file_{k}'] = v
                    
                    all_rows.append(row)
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")

    df = pd.DataFrame(all_rows)
    return df, file_info

# Sidebar
st.sidebar.header("📁 Data & Filters")
data_dir = st.sidebar.text_input("Benchmark Directory", "./benchmarks")
df, file_info = load_data(data_dir)

if df.empty:
    st.warning("No data loaded. Check the directory and JSON format.")
    st.stop()

# Filters
repos = st.sidebar.multiselect("Repositories", options=sorted(df['repo'].unique()), default=sorted(df['repo'].unique()))
tactics = st.sidebar.multiselect("Tactics", options=sorted(df['tactic'].unique()), default=sorted(df['tactic'].unique()))
df = df[(df['repo'].isin(repos)) & (df['tactic'].isin(tactics))]

# Add derived columns
df['file_ext'] = df['file'].str.extract(r'\.([^.]+)$')
df['nodes_changed'] = (df['full_nodes'] - df['incremental_nodes']).abs()
df['has_incremental'] = df['incremental_nodes'].notna() & (df['step'] >= 1)

# ======================
# 2. OVERVIEW METRICS
# ======================

st.title("🔍 CPG Incremental Parsing Dashboard")

# Metrics
total_files = len(df)
changed_files = len(df[(df['unchanged'] == False) & (df['step'] >= 1)])
incremental_success = len(df[df['has_incremental']])
unchanged_rate = df['unchanged'].mean() * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Files", f"{total_files:,}")
c2.metric("Files Changed", f"{changed_files:,}")
c3.metric("Incremental Success", f"{incremental_success:,}")
c4.metric("Unchanged Rate", f"{unchanged_rate:.1f}%")

st.divider()

# ======================
# 3. DATA QUALITY CHECK
# ======================

st.header("📊 Data Quality & Availability")

# Check timing fields
timing_cols = [c for c in df.columns if c.startswith('timing_')]
availability = {
    col.replace('timing_', ''): {
        'Available': df[col].notna().sum(),
        'Zero': (df[col] == 0).sum(),
        'Missing': df[col].isna().sum()
    }
    for col in timing_cols
}
avail_df = pd.DataFrame(availability).T
st.dataframe(avail_df, use_container_width=True)

# Highlight known issue
if df['timing_ts_full_parse_time_ms'].isna().any() and 'full_parse_time_ms' in df.columns:
    st.warning("""
    ⚠️ `ts_full_parse_time_ms` is missing for some files.  
    Falling back to `full_parse_time_ms` (legacy field) for full parse total.
    """)

# ======================
# 4. INCREMENTAL ANALYSIS
# ======================

st.header("🔄 Incremental Parsing Results")

# Fix: Use full_parse_time_ms if detailed timings are missing
# First, ensure we have the timing columns with default values
timing_columns = [
    'timing_ts_full_parse_time_ms', 'timing_cst_to_cpg_time_ms',
    'timing_ts_old_parse_time_ms', 'timing_text_diff_time_ms',
    'timing_ts_edit_apply_time_ms', 'timing_ts_incremental_parse_time_ms',
    'timing_cpg_incremental_update_time_ms'
]

for col in timing_columns:
    if col not in df.columns:
        df[col] = 0

# Compute full_total_ms robustly - fall back to legacy field if detailed timings missing
if 'full_parse_time_ms' in df.columns:
    df['full_total_ms'] = df['full_parse_time_ms'].fillna(0).astype(float)
    # If we have detailed timings and they're non-zero, prefer those
    detailed_full = (
        df['timing_ts_full_parse_time_ms'].fillna(0).astype(float) +
        df['timing_cst_to_cpg_time_ms'].fillna(0).astype(float)
    )
    # Use detailed timings where available and non-zero
    mask = detailed_full > 0
    df.loc[mask, 'full_total_ms'] = detailed_full.loc[mask]
else:
    df['full_total_ms'] = (
        df['timing_ts_full_parse_time_ms'].fillna(0).astype(float) +
        df['timing_cst_to_cpg_time_ms'].fillna(0).astype(float)
    )

# Compute incremental_total_ms from detailed timings
df['incremental_total_ms'] = (
    df['timing_ts_old_parse_time_ms'].fillna(0).astype(float) +
    df['timing_text_diff_time_ms'].fillna(0).astype(float) +
    df['timing_ts_edit_apply_time_ms'].fillna(0).astype(float) +
    df['timing_ts_incremental_parse_time_ms'].fillna(0).astype(float) +
    df['timing_cpg_incremental_update_time_ms'].fillna(0).astype(float)
)

# Only include files where both full and incremental are valid
valid = df[
    (df['full_total_ms'] > 0) &
    (df['incremental_total_ms'] > 0) &
    (df['has_incremental']) &
    (df['step'] >= 1)
].copy()

# Now compute speedup and time saved only for valid cases
if not valid.empty:
    valid['speedup'] = (valid['full_total_ms'] / valid['incremental_total_ms']).round(2)
    valid['time_saved_ms'] = (valid['full_total_ms'] - valid['incremental_total_ms']).round(1)
    
    # Add speedup back to main dataframe for time trend analysis
    df['speedup'] = None
    df['time_saved_ms'] = None
    # Copy computed values back to main df where they exist
    mask = df.index.isin(valid.index)
    df.loc[mask, 'speedup'] = valid['speedup']
    df.loc[mask, 'time_saved_ms'] = valid['time_saved_ms']
else:
    # Create empty columns for consistency
    df['speedup'] = None
    df['time_saved_ms'] = None

if valid.empty:
    st.warning("No valid incremental vs full comparisons found.")
else:
    st.success(f"Found {len(valid)} files with valid incremental data.")

    # Speedup distribution
    fig = px.histogram(valid, x='speedup', nbins=30, title="Speedup Distribution (Full / Incremental)")
    fig.add_vline(x=1, line_dash="dash", line_color="red", annotation_text="No speedup")
    st.plotly_chart(fig, use_container_width=True)

    # Scatter: Full vs Incremental
    fig = px.scatter(
        valid,
        x='full_total_ms',
        y='incremental_total_ms',
        size='full_nodes',
        color='repo',
        hover_data=['file', 'step', 'speedup'],
        title="Full vs Incremental Parse Time"
    )
    max_val = max(valid['full_total_ms'].max(), valid['incremental_total_ms'].max())
    fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(dash="dash", color="gray"))
    st.plotly_chart(fig, use_container_width=True)

    # Top speedups
    top_speedups = valid.nlargest(10, 'speedup')[
        ['file', 'step', 'repo', 'full_total_ms', 'incremental_total_ms', 'speedup', 'time_saved_ms']
    ]
    st.subheader("🏆 Top 10 Speedups")
    st.dataframe(top_speedups, use_container_width=True)

# ======================
# 5. CPG DIFF ANALYSIS
# ======================

st.header("🔍 CPG Structural Differences")

def parse_mismatch_details(detail_str: str) -> Dict[str, Any]:
    if not isinstance(detail_str, str):
        return {}
    # Extract function names and mismatch types
    functions = re.findall(r'function_name: "([^"]+)"', detail_str)
    details = re.findall(r'details: "([^"]+)"', detail_str)
    return {
        'num_functions': len(functions),
        'details': details
    }

mismatch_data = df[df['comparison_result'].str.contains("StructuralMismatch", na=False)].copy()
if mismatch_data.empty:
    st.info("No CPG mismatches found.")
else:
    mismatch_data['mismatch_info'] = mismatch_data['comparison_result'].apply(parse_mismatch_details)
    mismatch_data['num_mismatched_funcs'] = mismatch_data['mismatch_info'].apply(lambda x: x.get('num_functions', 0))

    st.metric("Files with CPG Mismatches", len(mismatch_data))
    st.dataframe(
        mismatch_data[['file', 'step', 'num_mismatched_funcs', 'comparison_result']].head(10),
        use_container_width=True
    )

# ======================
# 6. PERFORMANCE OVER TIME
# ======================

st.header("📈 Performance Over Time")

# Aggregate by step
time_trend = df.groupby(['repo', 'tactic', 'step']).agg(
    files_changed=('unchanged', lambda x: (x == False).sum()),
    avg_full_time=('full_total_ms', 'mean'),
    avg_incr_time=('incremental_total_ms', 'mean'),
    median_speedup=('speedup', 'median')
).reset_index()

fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=["Files Changed per Commit", "Avg Full Parse Time", "Median Speedup"],
    shared_xaxes=True, vertical_spacing=0.1
)

for (repo, tactic), group in time_trend.groupby(['repo', 'tactic']):
    label = f"{repo}/{tactic}"
    fig.add_trace(
        go.Scatter(x=group['step'], y=group['files_changed'], name=f"{label} - Changed Files", mode='lines+markers'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=group['step'], y=group['avg_full_time'], name=f"{label} - Full Time", mode='lines+markers'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=group['step'], y=group['median_speedup'], name=f"{label} - Speedup", mode='lines+markers'),
        row=3, col=1
    )

fig.update_layout(height=700, showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# ======================
# 7. FILE METRICS ANALYSIS
# ======================

st.header("📏 File Size & Change Analysis")

# Check if file metrics are available
file_metrics_cols = [c for c in df.columns if c.startswith('file_')]
if file_metrics_cols:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("File Size Distribution")
        if 'file_file_size_bytes' in df.columns:
            # Convert bytes to KB for better readability
            df['file_size_kb'] = df['file_file_size_bytes'] / 1024
            fig = px.histogram(
                df[df['file_size_kb'].notna()], 
                x='file_size_kb', 
                nbins=30, 
                title="File Size Distribution (KB)",
                labels={'file_size_kb': 'File Size (KB)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Line Count Distribution")
        if 'file_line_count' in df.columns:
            fig = px.histogram(
                df[df['file_line_count'].notna()], 
                x='file_line_count', 
                nbins=30, 
                title="Line Count Distribution",
                labels={'file_line_count': 'Lines of Code'}
            )
            st.plotly_chart(fig, use_container_width=True)

    # Change analysis for files with incremental data
    change_data = df[
        (df['file_changed_lines'].notna()) & 
        (df['file_proportion_lines_changed'].notna()) &
        (df['has_incremental'])
    ].copy()
    
    if not change_data.empty:
        st.subheader("Change Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                change_data,
                x='file_line_count',
                y='file_changed_lines',
                color='speedup',
                size='file_size_kb',
                hover_data=['file', 'step', 'file_proportion_lines_changed'],
                title="File Size vs Lines Changed",
                labels={
                    'file_line_count': 'Total Lines',
                    'file_changed_lines': 'Lines Changed'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                change_data,
                x='file_proportion_lines_changed',
                y='speedup',
                color='repo',
                size='file_size_kb',
                hover_data=['file', 'step', 'file_changed_lines'],
                title="Proportion Changed vs Speedup",
                labels={
                    'file_proportion_lines_changed': 'Proportion of Lines Changed',
                    'speedup': 'Speedup Factor'
                }
            )
            fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="No speedup")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Change Characteristics")
        change_summary = change_data.groupby(pd.cut(change_data['file_proportion_lines_changed'], bins=5)).agg({
            'speedup': ['mean', 'median', 'count'],
            'file_changed_lines': 'mean',
            'file_size_kb': 'mean'
        }).round(2)
        change_summary.columns = ['Avg Speedup', 'Median Speedup', 'Count', 'Avg Lines Changed', 'Avg File Size (KB)']
        st.dataframe(change_summary, use_container_width=True)
        
        # Top files by change proportion
        st.subheader("Files with Highest Change Proportion")
        top_changes = change_data.nlargest(10, 'file_proportion_lines_changed')[
            ['file', 'step', 'file_line_count', 'file_changed_lines', 'file_proportion_lines_changed', 'speedup']
        ]
        st.dataframe(top_changes, use_container_width=True)
    else:
        st.info("No change analysis data available. Files may be unchanged or missing metrics.")

# ======================
# 8. FILE-LEVEL INSIGHTS
# ======================

st.header("🗂️ File-Level Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Slowest Full Parses")
    slowest_full = df.nlargest(10, 'full_total_ms')[
        ['file', 'step', 'full_total_ms', 'full_nodes']
    ]
    st.dataframe(slowest_full, use_container_width=True)

with col2:
    st.subheader("Top 10 Slowest Incremental Parses")
    slowest_incr = df.nlargest(10, 'incremental_total_ms')[
        ['file', 'step', 'incremental_total_ms', 'speedup']
    ]
    st.dataframe(slowest_incr, use_container_width=True)

# Extension analysis
st.subheader("Performance by File Extension")
ext_perf = df.groupby('file_ext', observed=True).agg(
    files=('file', 'count'),
    avg_full_time=('full_total_ms', 'mean'),
    avg_incr_time=('incremental_total_ms', 'mean'),
    avg_speedup=('speedup', 'mean')
).round(2).sort_values('files', ascending=False)
st.dataframe(ext_perf, use_container_width=True)

# ======================
# 9. DEBUG / ERROR INSIGHTS
# ======================

st.header("🐞 Debug & Errors")

error_files = df[df['comparison_result'].str.contains("Failed", na=False)]
if not error_files.empty:
    st.warning(f"Found {len(error_files)} files with errors.")
    st.dataframe(error_files[['file', 'step', 'comparison_result']], use_container_width=True)

# ======================
# 10. RECOMMENDATIONS
# ======================

st.sidebar.divider()
st.sidebar.markdown("### 💡 Recommendations")

if incremental_success < 10:
    st.sidebar.warning("Few incremental successes — try a more active repo (e.g. tree-sitter)")

if valid['speedup'].median() and valid['speedup'].median() < 1.5:
    st.sidebar.info("Low speedup — check if edits are too large for effective incremental parsing")

if df['timing_ts_full_parse_time_ms'].isna().mean() > 0.3:
    st.sidebar.warning("Many missing detailed timings — fix Rust timing merge logic")