# 🎨 Welcome to DataCraft - Where Data Cleaning Meets Fun! 🎉
# Warning: This code may cause excessive organization and data cleanliness
# Side effects may include: improved data quality and occasional bursts of joy

import streamlit as st  # Our magical UI wizard ✨
import pandas as pd    # The data wrangling panda 🐼
import numpy as np     # Numbers go brrr 🔢
import base64         # For those sneaky encoding tricks 🕵️
import plotly.express as px      # Making charts look fancy AF 📊
import plotly.graph_objects as go # When regular charts just won't cut it 💅
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # The normalizing ninjas 🥷
import re             # Regex: Because sometimes string parsing feels like dark magic 🪄
from io import StringIO  # The string whisperer 🤫
from sklearn.impute import SimpleImputer  # The missing data detective 🔍
from sklearn.ensemble import IsolationForest  # The outlier bouncer 🚫
import textacy.preprocessing as tprep  # Text cleaning superhero 📝

# ---------------------------------------------
# 🎨 CSS & JS Magic Zone - Enter if you dare!
# Warning: Contains enough gradients to make a unicorn jealous
custom_css = """
<style>
    /* Particle canvas container */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        pointer-events: none;
    }
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        --dark-bg: #0f172a;
        --card-bg: rgba(15, 23, 42, 0.7);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --accent: #818cf8;
        --shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
    }
    .stApp {
        background: var(--dark-bg);
        color: var(--text-primary);
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        line-height: 1.6;
    }
    .title-container {
        position: relative;
        border-radius: 0 0 30px 30px;
        margin-bottom: 3rem;
        min-height: 300px;
    }
    .title-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 300px;
        z-index: 3;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        background: rgba(15, 23, 42, 0.4);
    }
    .upload-area {
        margin: 2rem 0;
        text-align: center;
    }
    .drag-drop-zone {
        border: 2px dashed var(--accent);
        border-radius: 12px;
        padding: 2rem;
        background: rgba(15, 23, 42, 0.7);
        transition: border 0.3s;
        cursor: pointer;
        position: relative;
    }
    .features {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
        padding: 0 1rem;
    }
    .feature-card {
        flex: 1;
        padding: 1.5rem;
        background: rgba(15, 23, 42, 0.7);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .stDataFrame {
        background: rgba(15, 23, 42, 0.7) !important;
        padding: 1rem !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    .dataframe {
        width: 100% !important;
        color: var(--text-primary) !important;
        background: transparent !important;
    }
    .dataframe th {
        background: rgba(129, 140, 248, 0.1) !important;
        padding: 0.75rem 1rem !important;
        font-weight: 600 !important;
        color: var(--accent) !important;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1) !important;
    }
    .dataframe td {
        padding: 0.75rem 1rem !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        color: var(--text-primary) !important;
    }
    .stSidebar {
        background: var(--card-bg) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    .stExpander {
        background: var(--card-bg) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        margin: 1rem 0 !important;
    }
    .plotly-chart {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    .stDownloadButton>button {
        width: 100% !important;
        background: var(--primary-gradient) !important;
        border: none !important;
    }
</style>
<div class="particles">
    <canvas id="particle-canvas"></canvas>
</div>
<script>
(function() {
    const canvas = document.getElementById('particle-canvas');
    const ctx = canvas.getContext('2d');
    let particles = [];
    const rowCount = 15;
    const colCount = 40;
    const waveSpeed = 0.02;
    const waveAmplitude = 20;
    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    function createParticles() {
        particles = [];
        const xSpacing = canvas.width / (colCount - 1);
        const ySpacing = canvas.height / (rowCount - 1);
        for(let row = 0; row < rowCount; row++) {
            for(let col = 0; col < colCount; col++) {
                particles.push({
                    x: col * xSpacing,
                    baseY: row * ySpacing,
                    angle: (row + col) * 0.5,
                    size: 1
                });
            }
        }
    }
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.angle += waveSpeed;
            const wave = Math.sin(p.angle) * waveAmplitude;
            const currentY = p.baseY + wave;
            const relativeY = currentY / canvas.height;
            const opacity = 0.2 + relativeY * 0.8;
            ctx.beginPath();
            ctx.arc(p.x, currentY, p.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
            ctx.fill();
        });
        requestAnimationFrame(draw);
    }
    window.addEventListener('resize', () => {
        resize();
        createParticles();
    });
    resize();
    createParticles();
    draw();
})();
</script>
"""

# ---------------------------------------------
# 🎪 The Main Show Setup
st.set_page_config(
    page_title="DataCraft",  # New name, same awesome app!
    page_icon="✨",         # Sparkles because we're fancy like that
    layout="wide",          # Go wide or go home
    initial_sidebar_state="expanded",  # Show off that sidebar
    menu_items={'Get Help': None, 'Report a bug': None, 'About': None}  # We live dangerously
)
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------------------------
# Title Section
st.markdown("""
<div class="title-container">
    <div class="title-overlay">
        <h1 style="
            font-size: 4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            line-height: 1.2;
        ">
            ✨ DataCraft Pro
        </h1>
        <p style="
            font-size: 1.5rem;
            color: #94a3b8;
            max-width: 800px;
            margin: 0 auto;
        ">
            Your Advanced Data Cleaning and Analysis Companion
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------
# File Upload Section
uploaded_files = st.file_uploader(
    "Upload CSV files to process",
    type=["csv"],
    accept_multiple_files=True,
    key="file_uploader"
)

dataframes = []
file_names = []
if uploaded_files:
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            file_names.append(file.name)
        except Exception as e:
            st.error(f"Error reading {file.name}: {str(e)}")

# Initialize processed_dfs and default options so they're defined even if no files are uploaded
processed_dfs = []
visualize = False
show_stats = False
clean_na = False
clean_duplicates = False
text_clean = False
norm_method = "None"
outlier_detection = False
dtype_conversion = False

# ---------------------------------------------
# Sidebar Controls for Processing (only if files are uploaded)
if uploaded_files and dataframes:
    with st.sidebar:
        st.markdown("""
        <div style="background: var(--card-bg); padding: 1rem; border-radius: 12px;">
            <h3 style="color: var(--accent);">⚙️ Processing Controls</h3>
        </div>
        """, unsafe_allow_html=True)
        # Merge Files (only enabled if more than one file is uploaded)
        merge_files = False
        if len(dataframes) > 1:
            merge_files = st.checkbox("Merge files", key="merge_files_sidebar")
            merge_type = st.selectbox(
                "Join Type",
                ["inner", "outer", "left", "right"],
                disabled=not merge_files,
                key="merge_type"
            )
        else:
            st.markdown("_Upload multiple files to enable merging_")
        # Data cleaning options
        clean_na = st.checkbox("Remove N/A values", help="Handle missing values", key="clean_na_sidebar")
        if clean_na:
            na_strategy = st.selectbox(
                "Missing Value Strategy",
                options=["remove", "mean", "median", "mode", "forward_fill", "backward_fill"],
                index=0,
                key="na_strategy"
            )
        clean_duplicates = st.checkbox("Remove duplicate rows", key="clean_duplicates_sidebar")
        text_clean = st.checkbox("Clean text data", key="text_clean_sidebar")
        # Transformation options
        norm_method = st.selectbox(
            "Scaling Method",
            ["None", "Min-Max", "Standard"],
            index=0,
            key="norm_method"
        )
        outlier_detection = st.checkbox("Remove outliers", key="outlier_detection")
        dtype_conversion = st.checkbox("Convert data types", key="convert_types_sidebar")
        if dtype_conversion:
            col_to_convert = st.selectbox("Column to convert", dataframes[0].columns, key="col_to_convert")
            new_type = st.selectbox("New data type", ["str", "int", "float", "datetime"], key="new_type")
        # Visualization and Statistics options
        visualize = st.checkbox("Enable Visualizations", key="visualize_sidebar")
        show_stats = st.checkbox("Show statistics", key="show_stats_sidebar")

# ---------------------------------------------
# 🧙‍♂️ Helper Functions - The Real MVPs
def clean_text_data(df):
    """
    The text cleaning spa treatment 💆‍♂️
    Your text comes in dirty, leaves clean and refreshed!
    """
    text_cols = df.select_dtypes(include='object').columns
    for col in text_cols:
        df[col] = df[col].apply(lambda x: tprep.normalize.whitespace(str(x)))
        df[col] = df[col].apply(tprep.remove.punctuation)
        df[col] = df[col].str.lower()
    return df

def normalize_data(df, method):
    """
    The data normalizing smoothie maker 🥤
    Blending those numbers into a consistent range!
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    if method == 'Min-Max':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def handle_missing_values(df, strategy='remove'):
    """
    Missing value whisperer 👻
    Making NaN problems disappear since 2024!
    """
    initial_len = len(df)
    if strategy == 'remove':
        df = df.dropna(how='any').reset_index(drop=True)
    elif strategy in ['mean', 'median', 'most_frequent']:
        imputer = SimpleImputer(strategy=strategy)
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    elif strategy == 'mode':
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None, inplace=True)
    elif strategy == 'forward_fill':
        df.ffill(inplace=True)
    elif strategy == 'backward_fill':
        df.bfill(inplace=True)
    rows_affected = initial_len - len(df) if strategy == 'remove' else df.isna().sum().sum()
    return df, rows_affected

def detect_outliers(df):
    """
    The outlier bouncer 🚪
    If your data point doesn't fit in, it ain't getting in!
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    clf = IsolationForest(contamination=0.05)
    df['is_outlier'] = clf.fit_predict(df[numeric_cols])
    return df[df['is_outlier'] == 1].drop('is_outlier', axis=1)

# ---------------------------------------------
# Data Processing Pipeline (only if files are uploaded)
if uploaded_files and dataframes:
    st.markdown("## 🔄 Processing Data in Real Time")
    for idx, df in enumerate(dataframes):
        df = df.copy()
        st.markdown(f"### Processing **{file_names[idx]}**")
        # Handle Missing Values
        if clean_na:
            df, rows_affected = handle_missing_values(df, strategy=na_strategy)
            st.info(f"Applied missing value strategy **{na_strategy}**. Remaining missing values: **{df.isna().sum().sum()}**")
        # Remove Duplicates
        if clean_duplicates:
            initial_len = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            st.info(f"Removed **{initial_len - len(df)}** duplicate rows")
        # Clean Text Data
        if text_clean:
            df = clean_text_data(df)
            st.info("Cleaned text data")
        # Normalize/Standardize Data
        if norm_method != "None":
            df = normalize_data(df, method=norm_method)
            st.info(f"Applied **{norm_method}** scaling")
        # Remove Outliers
        if outlier_detection:
            initial_len = len(df)
            df = detect_outliers(df)
            st.info(f"Removed **{initial_len - len(df)}** outlier rows")
        # Data Type Conversion
        if dtype_conversion and col_to_convert in df.columns:
            try:
                if new_type == "datetime":
                    df[col_to_convert] = pd.to_datetime(df[col_to_convert])
                else:
                    df[col_to_convert] = df[col_to_convert].astype(new_type)
                st.info(f"Converted column **{col_to_convert}** to **{new_type}**")
            except Exception as e:
                st.error(f"Failed to convert {col_to_convert}: {str(e)}")
        processed_dfs.append(df)
    # Merge files if selected and more than one file is processed
    if merge_files and len(processed_dfs) > 1:
        try:
            merged_df = processed_dfs[0]
            for df in processed_dfs[1:]:
                common_cols = list(set(merged_df.columns) & set(df.columns))
                merged_df = pd.merge(merged_df, df, how=merge_type, on=common_cols)
            processed_dfs = [merged_df]
            st.success(f"Successfully merged {len(dataframes)} files using **{merge_type}** join")
        except Exception as e:
            st.error(f"Merge failed: {str(e)}")

# ---------------------------------------------
# Download Processed Data (always reflecting the latest processed table)
if processed_dfs:
    if len(processed_dfs) > 1:
        download_choice = st.selectbox("Select file to download", options=[f"{i+1}: {file_names[i]}" for i in range(len(file_names))])
        download_index = int(download_choice.split(":")[0]) - 1
    else:
        download_index = 0
    csv = processed_dfs[download_index].to_csv(index=False).encode()
    st.download_button(
        label="📥 Download Processed Data",
        data=csv,
        file_name='processed_data.csv',
        mime='text/csv',
        help="Download the cleaned and processed dataset",
        key='download-csv'
    )

# ---------------------------------------------
# Final Display Section
# If any transformation was applied, show the final processed data table at the bottom;
# otherwise, display the original Data Overview (via a checkbox).
if uploaded_files and dataframes:
    transformations_applied = clean_na or clean_duplicates or text_clean or (norm_method != "None") or outlier_detection or dtype_conversion
    if transformations_applied:
        st.markdown("## 🎉 Final Processed Data")
        for i, df in enumerate(processed_dfs):
            display_name = file_names[i] if i < len(file_names) else f"DataFrame {i+1}"
            st.markdown(f'''
                <div style="margin-top:1rem; margin-bottom:0.5rem; font-weight:bold;">
                    Processed Data: {display_name}
                    <span style="float: right; font-size: 0.8em; color: #4ade80;">
                        {df.shape[0]} rows × {df.shape[1]} columns
                    </span>
                </div>
            ''', unsafe_allow_html=True)
            with st.container():
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.metric("Total Values", df.size)
                    st.metric("Remaining Missing", int(df.isna().sum().sum()))
                with col2:
                    st.dataframe(
                        df.style.applymap(lambda x: 'background-color: #ff6b6b' if pd.isna(x) else ''),
                        height=600,
                        use_container_width=True
                    )
    else:
        if st.checkbox("Show Data Overview", value=True, key="show_overview"):
            st.markdown("## Data Overview")
            for i, df in enumerate(dataframes):
                display_name = file_names[i] if i < len(file_names) else f"DataFrame {i+1}"
                st.markdown(f'''
                    <div style="margin-top:1rem; margin-bottom:0.5rem; font-weight:bold;">
                        DataFrame: {display_name}
                        <span style="float: right; font-size: 0.8em; color: #4ade80;">
                            {df.shape[0]} rows × {df.shape[1]} columns
                        </span>
                    </div>
                ''', unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True)

# ---------------------------------------------
# Advanced Statistics & Visualizations
if processed_dfs and show_stats:
    with st.expander("📈 Advanced Statistics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Descriptive Statistics")
            stats_df = processed_dfs[0].describe().astype(str)
            st.dataframe(stats_df, use_container_width=True)
        with col2:
            st.markdown("### Data Overview")
            overview = pd.DataFrame({
                'Data Type': processed_dfs[0].dtypes.astype(str),
                'Missing Values': processed_dfs[0].isna().sum().astype(str)
            })
            st.dataframe(overview, use_container_width=True)

if processed_dfs and visualize:
    st.markdown("## 📊 Interactive Visualizations")
    plot_col, settings_col = st.columns([3, 1])
    df_viz = processed_dfs[0]
    with settings_col:
        plot_type = st.selectbox(
            "Chart Type",
            ["Histogram", "Box Plot", "Scatter", "Line", "3D Scatter", "Violin", "Heatmap", "Bar"],
            key="plot_type"
        )
        color_theme = st.color_picker("Chart Color", "#1f77b4", key="chart_color")
        numeric_cols = df_viz.select_dtypes(include=np.number).columns
        x_axis = st.selectbox("X Axis", numeric_cols, key="x_axis")
        if plot_type in ["Scatter", "Line", "Bar"]:
            y_axis = st.selectbox("Y Axis", numeric_cols, key="y_axis")
        if plot_type == "3D Scatter":
            y_axis = st.selectbox("Y Axis", numeric_cols, key="y_axis_3d")
            z_axis = st.selectbox("Z Axis", numeric_cols, key="z_axis")
        if plot_type in ["Scatter", "Line", "3D Scatter"]:
            marker_size = st.slider("Marker Size", 1, 20, 8, key="marker_size")
        if plot_type in ["Histogram", "Bar"]:
            bin_count = st.slider("Number of Bins", 5, 100, 30, key="bin_count")
    with plot_col:
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            dragmode='pan',
            showlegend=True,
            hovermode='closest'
        )
        try:
            if plot_type == "Histogram":
                fig = px.histogram(df_viz, x=x_axis, nbins=bin_count, color_discrete_sequence=[color_theme])
            elif plot_type == "Scatter":
                fig = px.scatter(df_viz, x=x_axis, y=y_axis, color_discrete_sequence=[color_theme])
                fig.update_traces(marker=dict(size=marker_size))
            elif plot_type == "Line":
                fig = px.line(df_viz, x=x_axis, y=y_axis, color_discrete_sequence=[color_theme])
            elif plot_type == "3D Scatter":
                fig = px.scatter_3d(df_viz, x=x_axis, y=y_axis, z=z_axis,
                                    color_discrete_sequence=[color_theme])
                fig.update_traces(marker=dict(size=marker_size))
            elif plot_type == "Bar":
                fig = px.bar(df_viz, x=x_axis, y=y_axis, color_discrete_sequence=[color_theme])
            elif plot_type == "Box":
                fig = px.box(df_viz, y=x_axis, color_discrete_sequence=[color_theme])
            elif plot_type == "Violin":
                fig = px.violin(df_viz, y=x_axis, box=True, color_discrete_sequence=[color_theme])
            elif plot_type == "Heatmap":
                corr_matrix = df_viz.select_dtypes(include=np.number).corr()
                fig = px.imshow(corr_matrix, color_continuous_scale='RdBu')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
