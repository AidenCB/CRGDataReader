import streamlit as st
import pandas as pd
import numpy as np

import base64
import io
import logging
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = APP_DIR / "images"
PAGE_ICON_PATH = ASSET_DIR / "RamapoArch.png"
MAX_UPLOAD_SIZE_MB = 200
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_UPLOAD_EXTENSIONS = (".csv", ".xlsx", ".xls", ".txt")

logger = logging.getLogger(__name__)

NOAA_PRESET = "NOAA daily climate"
GENERIC_PRESET = "Generic data"

NOAA_MEASUREMENT_COLUMNS = {
    "dapr", "dasf", "mdpr", "mdsf", "prcp", "snow", "snwd", "tavg", "tmax", "tmin", "tobs",
    "awnd", "wdf2", "wdf5", "wsf2", "wsf5", "wt01", "wt02",
    "wt03", "wt04", "wt05", "wt06", "wt07", "wt08", "wt09",
    "wt10", "wt11", "wt13", "wt14", "wt15", "wt16", "wt17",
    "wt18", "wt19", "wt21", "wt22"
}

PRESET_CONFIGS = {
    NOAA_PRESET: {
        "default_dayfirst": False,
        "preferred_date_columns": ["date"],
        "repair_missing_leading_dates": True,
        "numeric_placeholder_columns": NOAA_MEASUREMENT_COLUMNS,
        "numeric_placeholders": {-999, 999, -9, 9999, -9999, -200},
    },
    GENERIC_PRESET: {
        "default_dayfirst": False,
        "preferred_date_columns": [],
        "repair_missing_leading_dates": False,
        "numeric_placeholder_columns": set(),
        "numeric_placeholders": set(),
    },
}

STATE_RAW_DF = "dfRaw"
STATE_WORKING_DF = "workingDf"
STATE_FILENAME = "filename"
STATE_UPLOAD_CONFIG = "upload_config"
STATE_PRESET = "data_preset"
STATE_DATE_DAYFIRST = "date_dayfirst"
STATE_PERCENTILE_DF = "dfPercent"
STATE_PERCENTILE_VALUE = "userPercent"
STATE_WAVELET_PNG = "wavelet_png"
STATE_WAVELET_CONFIG = "wavelet_config"
STATE_WAVELET_FILENAME = "wavelet_default_filename"

DERIVED_STATE_KEYS = [
    STATE_WORKING_DF,
    STATE_PERCENTILE_DF,
    STATE_PERCENTILE_VALUE,
    STATE_WAVELET_PNG,
    STATE_WAVELET_CONFIG,
    STATE_WAVELET_FILENAME,
]

def main():
    st.set_page_config(
        page_title="DataReader Web",
        layout="wide",
        page_icon=str(PAGE_ICON_PATH) if PAGE_ICON_PATH.exists() else None,
    )
    render_app_header()

    upload_options = render_upload_controls()
    uploadedFile = upload_options["uploaded_file"]

    if uploadedFile is None:
        return

    try:
        validate_uploaded_file(uploadedFile, upload_options["separator"])
    except ValueError as e:
        st.error(str(e))
        return

    upload_config = build_upload_config(uploadedFile, upload_options)
    if st.session_state.get(STATE_UPLOAD_CONFIG) != upload_config:
        reset_loaded_state()
        st.session_state[STATE_UPLOAD_CONFIG] = upload_config

    should_refresh = st.button("Refresh Data")
    if st.session_state.get(STATE_WORKING_DF) is None or should_refresh:
        try:
            dfRaw = read_uploaded_file(uploadedFile, upload_options["separator"])
            workingDf = cleanData(
                dfRaw,
                preset_name=upload_options["preset_name"],
                date_dayfirst=upload_options["date_dayfirst"]
            )

            st.session_state[STATE_FILENAME] = safeFilename(uploadedFile.name, default="data", strip_extension=True)
            st.session_state[STATE_RAW_DF] = dfRaw.copy()
            st.session_state[STATE_WORKING_DF] = workingDf
            st.success("File uploaded successfully.")

        except Exception as e:
            logger.exception("Failed to read uploaded file %s", getattr(uploadedFile, "name", "unknown"))
            st.error("The file could not be read. Check the file format and separator, then try again.")
            st.session_state[STATE_WORKING_DF] = None

    workingDf = st.session_state.get(STATE_WORKING_DF)
    if workingDf is None:
        st.warning("No data loaded. Please upload a file.")
        return

    mainMenu = st.selectbox(
        "Select an option to analyze or edit data",
        [
            "View Data",
            "Statistics",
            "View/Change Datatypes",
            "Reset Data",
            "Edit Data",
            "Visualize Data",
            "Save / Export"
        ],
        key="mainMenu",
        label_visibility="collapsed",
    )

    if mainMenu != "View Data":
        with st.expander("View Dataframe (10 rows)"):
            st.dataframe(workingDf.head(10))

    if mainMenu == "View Data":
        render_view_data(workingDf)
    elif mainMenu == "Statistics":
        render_statistics(workingDf)
    elif mainMenu == "View/Change Datatypes":
        render_datatypes(workingDf)
    elif mainMenu == "Edit Data":
        render_edit_data(workingDf)
    elif mainMenu == "Visualize Data":
        renderWaveletSection(workingDf)
    elif mainMenu == "Reset Data":
        render_reset(upload_options["preset_name"], upload_options["date_dayfirst"])
    elif mainMenu == "Save / Export":
        render_export(st.session_state.get(STATE_WORKING_DF, workingDf))

def render_app_header():
    logo_uri = imageDataUri(ASSET_DIR / "whiteRamapoLogo.png")
    logo_html = f'<img src="{logo_uri}" alt="Ramapo College logo" />' if logo_uri else '<strong>RAMAPO COLLEGE</strong>'

    st.markdown("""
        <style>
        :root {
            --ramapo-maroon: #862633;
            --ramapo-maroon-dark: #651b27;
            --ramapo-red: #C41E1E;
            --ramapo-ink: #25282A;
            --ramapo-paper: #F7F2F2;
            --ramapo-panel: #FFFFFF;
            --ramapo-border: #D9C8CA;
        }

        [data-testid="stHeader"] {
            display: none;
        }

        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--ramapo-paper);
            color: var(--ramapo-ink);
        }

        [data-testid="stSidebar"] {
            background-color: var(--ramapo-panel);
        }

        .rdw-hero {
            display: flex;
            align-items: center;
            gap: 2rem;
            background: var(--ramapo-maroon);
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            border: 1px solid var(--ramapo-maroon-dark);
            box-shadow: 0 8px 24px rgba(37, 40, 42, 0.12);
            margin-bottom: 1.5rem;
        }

        .rdw-hero img {
            width: 220px;
            max-width: 28vw;
            height: auto;
            display: block;
        }

        .rdw-hero h1 {
            color: #FFFFFF;
            font-size: 2rem;
            line-height: 1.15;
            margin: 0 0 0.35rem 0;
            font-weight: 800;
            letter-spacing: 0;
        }

        .rdw-hero p {
            color: #F2E6E8 !important;
            font-size: 1rem;
            margin: 0;
        }

        .rdw-hero .rdw-logo strong {
            color: #FFFFFF;
            font-size: 1.3rem;
            letter-spacing: 0;
        }

        label, [data-testid="stMarkdownContainer"] p, .stMarkdown {
            color: var(--ramapo-ink);
        }

        div[data-baseweb="select"] > div,
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input {
            background-color: var(--ramapo-panel) !important;
            color: var(--ramapo-ink) !important;
            border: 1px solid var(--ramapo-border) !important;
            border-radius: 6px;
            min-height: 2.75rem;
        }

        div[data-baseweb="select"] > div:hover,
        [data-testid="stTextInput"] input:hover,
        [data-testid="stNumberInput"] input:hover {
            border-color: var(--ramapo-maroon) !important;
        }

        div[data-baseweb="select"] svg {
            fill: var(--ramapo-maroon) !important;
        }

        [data-testid="stFileUploader"] {
            background-color: #9B2D3D;
            max-width: 1200px;
            padding: 0.75rem;
            border-radius: 8px;
            border: 1px solid var(--ramapo-red);
        }

        [data-testid="stFileUploader"] > div {
            display: flex; 
            background-color: var(--ramapo-panel) !important;
            border-radius: 6px;
            padding: 1rem 1.25rem !important;
            gap: 2rem;
            width: 100%;
            color: var(--ramapo-ink) !important;
            align-items: center;
        }

        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] p {
            color: var(--ramapo-ink) !important;
        }

        .stButton > button {
            background-color: var(--ramapo-maroon);
            color: #FFFFFF;
            border: none;
            border-radius: 6px;
            padding: 0.55em 1em;
            font-weight: 600;
        }

        .stButton > button:hover {
            background-color: var(--ramapo-maroon-dark);
            border: none;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--ramapo-border);
            border-radius: 6px;
            overflow: hidden;
            background: var(--ramapo-panel);
        }

        [data-testid="stExpander"] {
            background: var(--ramapo-panel);
            border: 1px solid var(--ramapo-border);
            border-radius: 6px;
        }

        [data-testid="stElementToolbar"] {
            display: none !important;
        }

        @media (max-width: 760px) {
            .rdw-hero {
                align-items: flex-start;
                flex-direction: column;
                gap: 1rem;
            }

            .rdw-hero img {
                width: 190px;
                max-width: 80vw;
            }

            .rdw-hero h1 {
                font-size: 1.65rem;
            }
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="rdw-hero">
            <div class="rdw-logo">{logo_html}</div>
            <div>
                <h1>DataReader Web</h1>
                <p>Created by Aiden Cabrera for the Ramapo Climate Research Group</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_upload_controls():
    preset_name = st.selectbox(
        "Data preset",
        list(PRESET_CONFIGS.keys()),
        index=0,
        key=STATE_PRESET
    )
    preset_config = get_preset_config(preset_name)

    date_dayfirst = preset_config["default_dayfirst"]
    st.session_state[STATE_DATE_DAYFIRST] = date_dayfirst

    uploadedFile = st.file_uploader("Upload CSV, Excel, or TXT file", type=["csv", "xlsx", "xls", "txt"])

    actual_sep = ","
    sep_choice = "Comma (,)"
    custom_sep = ","

    if uploadedFile is not None and is_delimited_file(uploadedFile.name):
        sep_mapping = {
            "Comma (,)": ",",
            "Whitespace (multiple spaces/tabs)": r"\s+",
            "Tab (\\t)": "\t",
            "Semicolon (;)": ";",
            "Exact Space ( )": " ",
            "Custom": "custom"
        }
        sep_choice = st.selectbox("Select separator", list(sep_mapping.keys()), key="separator_choice")
        actual_sep = sep_mapping[sep_choice]
        if actual_sep == "custom":
            custom_sep = st.text_input("Enter custom separator", value=",", key="custom_separator")
            actual_sep = custom_sep

    return {
        "uploaded_file": uploadedFile,
        "preset_name": preset_name,
        "date_dayfirst": date_dayfirst,
        "separator": actual_sep,
        "separator_choice": sep_choice,
        "custom_separator": custom_sep,
    }

def render_view_data(workingDf):
    row_count = workingDf.shape[0]
    if row_count == 0:
        st.info("The loaded data has columns but no rows.")
        st.dataframe(workingDf)
        return

    numShow = st.number_input(
        f"Rows to show (1-{row_count})",
        min_value=1,
        max_value=row_count,
        value=min(10, row_count)
    )
    st.dataframe(view_head(workingDf, numShow))

def render_statistics(workingDf):
    catColNames = workingDf.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    numericCols = workingDf.select_dtypes(include=['number']).columns.tolist()

    available_stat_types = []
    if numericCols:
        available_stat_types.append("Numeric")
    if catColNames:
        available_stat_types.append("Categorical")

    if not available_stat_types:
        st.warning("No numeric or categorical columns found.")
        return

    statTypeChoice = st.selectbox("Choose statistics type", available_stat_types, key="stat_type_choice")

    if statTypeChoice == "Numeric":
        render_numeric_statistics(workingDf, numericCols)
    elif statTypeChoice == "Categorical":
        render_categorical_statistics(workingDf, catColNames)

def render_numeric_statistics(workingDf, numericCols):
    st.subheader("Numeric Statistics")
    dfStats = showMathInfo(workingDf)

    if dfStats is None:
        st.warning("No numeric columns found in the dataset.")
        return

    numStatOption = st.radio(
        "Choose a numeric statistic to view",
        ["Count per column", "Mean", "Std", "Min/Max rows", "Percentile", "All statistics"],
        key="num_stat_radio"
    )

    if numStatOption == "Count per column":
        render_statistic_row(dfStats, numericCols, "count", "Count", "_countPerColumn")
    elif numStatOption == "Mean":
        render_statistic_row(dfStats, numericCols, "mean", "Mean", "_meanPerColumn")
    elif numStatOption == "Std":
        render_statistic_row(dfStats, numericCols, "std", "Standard deviation", "_stdPerColumn")
    elif numStatOption == "Min/Max rows":
        render_min_max_rows(workingDf, dfStats, numericCols)
    elif numStatOption == "Percentile":
        render_percentile_tools(workingDf)
    elif numStatOption == "All statistics":
        st.dataframe(dfStats)

def render_statistic_row(dfStats, numericCols, stat_key, label, suffix):
    colChoice = st.selectbox("Select column or choose All", ["All"] + numericCols, key=f"{stat_key}_col")
    if colChoice == "All":
        st.write(f"{label} per column:")
        st.write(dfStats.loc[stat_key])
        saveStatistics(dfStats.loc[[stat_key]], suffix, widget_prefix=stat_key)
    else:
        st.write(f"{label} for column '{colChoice}': {dfStats.at[stat_key, colChoice]}")
        saveStatistics(
            pd.DataFrame({colChoice: [dfStats.at[stat_key, colChoice]]}),
            f"_{colChoice}_{stat_key}",
            widget_prefix=f"{stat_key}_single"
        )

def render_min_max_rows(workingDf, dfStats, numericCols):
    colChoice = st.selectbox("Select column for min/max or choose All", ["All"] + numericCols, key="minmax_col")
    columns_to_show = numericCols if colChoice == "All" else [colChoice]

    for column in columns_to_show:
        minVal = dfStats.at['min', column]
        maxVal = dfStats.at['max', column]
        minRow = workingDf[workingDf[column] == minVal]
        maxRow = workingDf[workingDf[column] == maxVal]
        st.write(f"Column: {column}")
        st.write("Min row(s):")
        st.dataframe(minRow)
        st.write("Max row(s):")
        st.dataframe(maxRow)

def render_percentile_tools(workingDf):
    userPercent = st.number_input(
        "Percentile (0-100)", min_value=0.0, max_value=100.0, value=50.0, step=0.1
    )

    if st.button("Show Percentile"):
        percent = round(userPercent / 100.0, 4)
        st.session_state[STATE_PERCENTILE_DF] = workingDf.quantile(percent, numeric_only=True)
        st.session_state[STATE_PERCENTILE_VALUE] = userPercent

    if STATE_PERCENTILE_DF not in st.session_state:
        return

    dfPercent = st.session_state[STATE_PERCENTILE_DF]
    pctLabel = st.session_state[STATE_PERCENTILE_VALUE]

    st.write(f"**{pctLabel}th percentile thresholds:**")
    thresholdDf = pd.DataFrame({
        "Column": dfPercent.index,
        f"{pctLabel}th Percentile Cutoff": dfPercent.values
    })
    st.dataframe(thresholdDf, hide_index=True)

    choice = st.radio(
        "Filter rows relative to percentile:",
        ("None", "At or below percentile", "At or above percentile"),
        key="percentile_choice"
    )

    if choice == "None":
        return

    filterCol = st.selectbox(
        "Select column to filter by",
        dfPercent.index.tolist(),
        key="percentile_filter_col"
    )

    filtered, threshold = findPercentiles(workingDf, dfPercent, choice, selected_col=filterCol)
    direction = "at or below" if choice == "At or below percentile" else "at or above"
    st.write(f"**{len(filtered)} rows** where `{filterCol}` is {direction} **{threshold:.4f}** ({pctLabel}th percentile)")
    st.dataframe(filtered)

    if not filtered.empty:
        csv = filtered.to_csv(index=False).encode('utf-8')
        defaultName = downloadFilename(
            f"{st.session_state.get(STATE_FILENAME, 'data')}_{filterCol}_{choice.replace(' ', '_')}_{pctLabel}pct",
            "csv",
            default="filtered_data"
        )
        exportName = st.text_input("Export filename", defaultName, max_chars=60, key="pctl_export_name")
        if st.button("Keep filtered rows as working data", key="pctl_keep_filtered"):
            set_working_df(filtered.reset_index(drop=True))
            st.success(f"Working data replaced with {len(filtered)} filtered rows.")
            st.dataframe(st.session_state[STATE_WORKING_DF].head())

        st.download_button(
            label=f"Download filtered data ({len(filtered)} rows)",
            data=csv,
            file_name=downloadFilename(exportName, "csv", default="filtered_data"),
            mime="text/csv",
            key="pctl_download"
        )

def render_categorical_statistics(workingDf, catColNames):
    st.subheader("Categorical Statistics")
    catStats = showCategoricalInfo(workingDf)

    if catStats is None:
        st.info("No categorical columns to display.")
        return

    catStatOption = st.radio(
        "Choose a categorical statistic to view",
        ["Unique counts", "Most frequent values", "Least frequent values", "All statistics"],
        key="cat_stat_radio"
    )

    if catStatOption == "Unique counts":
        resultDf = pd.DataFrame(workingDf[catColNames].nunique(), columns=["Unique Count"])
        st.write("Number of unique values per column:")
        st.dataframe(resultDf)
    elif catStatOption == "Most frequent values":
        colChoice = st.selectbox("Select column to view most frequent values", catColNames, key="cat_most_freq_col")
        resultDf = workingDf[colChoice].value_counts().head(10).reset_index()
        resultDf.columns = [colChoice, "Count"]
        st.write(f"Top 10 most frequent values for '{colChoice}':")
        st.dataframe(resultDf)
    elif catStatOption == "Least frequent values":
        colChoice = st.selectbox("Select column to view least frequent values", catColNames, key="cat_least_freq_col")
        resultDf = workingDf[colChoice].value_counts().tail(10).reset_index()
        resultDf.columns = [colChoice, "Count"]
        st.write(f"10 least frequent values for '{colChoice}':")
        st.dataframe(resultDf)
    elif catStatOption == "All statistics":
        st.write("Full categorical statistics:")
        st.dataframe(catStats)

def render_datatypes(workingDf):
    st.subheader("View/Change Datatypes")
    if workingDf.shape[1] == 0:
        st.info("No columns are available.")
        return

    st.write(workingDf.dtypes)

    colToChange = st.selectbox("Select column to change datatype", workingDf.columns.tolist(), key="dtype_col")
    dtypeChoice = st.selectbox("Target datatype", ["int", "float", "string", "date"], key="dtype_target")
    if st.button("Change datatype"):
        try:
            updated = convert_column_type(
                workingDf,
                colToChange,
                dtypeChoice,
                dayfirst=st.session_state.get(STATE_DATE_DAYFIRST, False)
            )
            set_working_df(updated)
            st.success(f"Converted {colToChange} to {dtypeChoice}")
        except Exception as e:
            st.error(str(e))

def render_edit_data(workingDf):
    st.subheader("Edit data")
    if workingDf.shape[1] == 0:
        st.info("No columns are available to edit.")
        return

    editAction = st.selectbox(
        "Choose edit action",
        ["Rename column", "Delete column", "Delete row", "Drop duplicates", "Sort by column", "Change Date"],
        key="edit_action"
    )

    if editAction == "Rename column":
        render_rename_column(workingDf)
    elif editAction == "Delete column":
        render_delete_columns(workingDf)
    elif editAction == "Delete row":
        render_delete_rows(workingDf)
    elif editAction == "Drop duplicates":
        render_drop_duplicates(workingDf)
    elif editAction == "Sort by column":
        render_sort_dataframe(workingDf)
    elif editAction == "Change Date":
        render_change_date(workingDf)

def render_rename_column(workingDf):
    colToEdit = st.selectbox("Select column to rename", workingDf.columns.tolist(), key="rename_col")
    newColName = st.text_input("New column name")
    if st.button("Rename column"):
        try:
            updated = rename_column(workingDf, colToEdit, newColName)
            set_working_df(updated)
            st.success(f"Renamed {colToEdit} to {newColName}")
            st.dataframe(updated.head())
        except Exception as e:
            st.error(str(e))

def render_delete_columns(workingDf):
    colsToDelete = []
    for col in workingDf.columns:
        if st.checkbox(f"Delete {col}", key=f"del_col_{col}"):
            colsToDelete.append(col)

    if st.button("Delete selected columns"):
        try:
            updated = delete_columns(workingDf, colsToDelete)
            set_working_df(updated)
            st.success(f"Deleted columns: {', '.join(colsToDelete)}")
            st.dataframe(updated.head())
        except Exception as e:
            st.error(str(e))

def render_delete_rows(workingDf):
    rowsToDelete = []
    userCond = st.radio("How to select rows?", options=["By condition", "By position"], key="del_row_method")

    if userCond == "By condition":
        columnCond = workingDf.select_dtypes(include=['number']).columns.tolist()
        if not columnCond:
            st.info("No numeric columns available for condition-based deletion.")
        else:
            userCol = st.radio("Select column to test against", columnCond, key="del_row_col")
            defaultValue = pd.to_numeric(workingDf[userCol], errors='coerce').mean()
            if pd.isna(defaultValue):
                defaultValue = 0.0
            userValue = st.number_input(
                f"Value to compare to '{userCol}' (the default is the average)",
                value=float(defaultValue)
            )
            condition = st.radio("Condition", ["Greater than", "Less than", "Equal to"], key="del_row_cond")
            rowsToDelete = rows_matching_condition(workingDf, userCol, condition, userValue)

    elif userCond == "By position":
        maxRow = workingDf.shape[0] - 1
        if maxRow >= 0:
            rowIndex = st.number_input(f"Row index to delete (0 to {maxRow})", min_value=0, max_value=maxRow, value=0)
            rowsToDelete.append(rowIndex)

    if rowsToDelete:
        st.write(f"Rows matching criteria: {len(rowsToDelete)}")
        if st.button("Delete selected rows"):
            try:
                updated = delete_rows(workingDf, rowsToDelete)
                set_working_df(updated)
                st.success(f"Deleted {len(rowsToDelete)} rows")
                st.dataframe(updated.head())
            except Exception as e:
                st.error(str(e))

def render_drop_duplicates(workingDf):
    subsetCols = st.multiselect("Subset columns for duplicate detection (empty = all columns)", workingDf.columns.tolist())
    keepOption = st.selectbox("Keep which", ["first", "last", "none"], key="dup_keep")
    if st.button("Drop duplicates"):
        before = len(workingDf)
        updated = drop_duplicate_rows(workingDf, subsetCols, keepOption)
        set_working_df(updated)
        st.success(f"Dropped duplicates, rows before: {before}, rows after: {len(updated)}")

def render_sort_dataframe(workingDf):
    sortCol = st.selectbox("Select column to sort by", workingDf.columns.tolist(), key="sort_col")
    ascending = st.checkbox("Sort ascending", value=True)
    if st.button("Sort"):
        updated = sort_dataframe(workingDf, sortCol, ascending)
        set_working_df(updated)
        st.success(f"Sorted by {sortCol} {'ascending' if ascending else 'descending'}")
        st.dataframe(updated.head())

def render_change_date(workingDf):
    dateCols = [col for col in workingDf.columns if pd.api.types.is_datetime64_any_dtype(workingDf[col])]
    if not dateCols:
        st.info("No datetime columns found. Convert a column to date type in the Datatypes menu first.")
        return

    colToEdit = st.selectbox("Select datetime column", dateCols, key="date_col")
    components = []
    if st.checkbox("Year", key="dt_year"):
        components.append('year')
    if st.checkbox("Month", key="dt_month"):
        components.append('month')
    if st.checkbox("Day", key="dt_day"):
        components.append('day')
    if st.checkbox("Hour", key="dt_hour"):
        components.append('hour')
    if st.checkbox("Minute", key="dt_min"):
        components.append('minute')
    if st.checkbox("Second", key="dt_sec"):
        components.append('second')

    if st.button("Apply date extraction"):
        if not components:
            st.warning("Select at least one component to extract.")
            return

        try:
            updated = extract_date_components(workingDf, colToEdit, components)
            set_working_df(updated)
            st.success("Datetime components extracted")
            st.dataframe(updated.head())
        except Exception as e:
            st.error(str(e))

def render_reset(preset_name, date_dayfirst):
    st.write("Use these actions to clean the working data. Changes are applied to the working copy.")

    if st.button("Reset working data to original upload"):
        if STATE_RAW_DF in st.session_state and st.session_state[STATE_RAW_DF] is not None:
            workingDf = cleanData(
                st.session_state[STATE_RAW_DF].copy(),
                preset_name=preset_name,
                date_dayfirst=date_dayfirst
            )
            set_working_df(workingDf)
            st.success("Reset complete - data re-cleaned from original upload.")
            st.dataframe(workingDf.head())
        else:
            st.warning("No original data found. Please re-upload your file.")

def render_export(workingDf):
    st.subheader("Save or export working data")

    defaultFilename = "export"
    if STATE_FILENAME in st.session_state:
        defaultFilename = f"{st.session_state[STATE_FILENAME]}_edited"

    newFilename = st.text_input("Filename for download", defaultFilename, max_chars=60)
    csv = workingDf.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=downloadFilename(newFilename, "csv", default="export"),
        mime="text/csv"
    )

def set_working_df(df):
    st.session_state[STATE_WORKING_DF] = df
    clear_derived_analysis_state()

def clear_derived_analysis_state():
    for key in [STATE_PERCENTILE_DF, STATE_PERCENTILE_VALUE, STATE_WAVELET_PNG, STATE_WAVELET_CONFIG, STATE_WAVELET_FILENAME]:
        st.session_state.pop(key, None)

def reset_loaded_state():
    for key in [STATE_RAW_DF, STATE_FILENAME] + DERIVED_STATE_KEYS:
        st.session_state.pop(key, None)

def build_upload_config(uploaded_file, upload_options):
    return (
        uploaded_file.name,
        getattr(uploaded_file, "size", None),
        upload_options["preset_name"],
        upload_options["separator"],
        upload_options["date_dayfirst"],
    )

def get_preset_config(preset_name):
    return PRESET_CONFIGS.get(preset_name, PRESET_CONFIGS[GENERIC_PRESET])

def is_delimited_file(filename):
    return str(filename).lower().endswith((".csv", ".txt"))

def validate_uploaded_file(uploaded_file, separator):
    filename = str(getattr(uploaded_file, "name", "")).strip()
    if not filename.lower().endswith(ALLOWED_UPLOAD_EXTENSIONS):
        allowed = ", ".join(ALLOWED_UPLOAD_EXTENSIONS)
        raise ValueError(f"Unsupported file type. Upload one of: {allowed}.")

    size = getattr(uploaded_file, "size", None)
    if size is not None and size > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError(f"File is too large. Maximum upload size is {MAX_UPLOAD_SIZE_MB} MB.")

    if is_delimited_file(filename) and separator is not None and str(separator) == "":
        raise ValueError("Separator cannot be empty.")

def read_uploaded_file(uploaded_file, separator):
    filename = str(uploaded_file.name).lower()
    uploaded_file.seek(0)

    if filename.endswith((".csv", ".txt")):
        try:
            return pd.read_csv(uploaded_file, header=None, sep=separator, engine='python')
        except pd.errors.EmptyDataError as e:
            raise ValueError("The uploaded file is empty.") from e
        except pd.errors.ParserError as e:
            raise ValueError("The uploaded file could not be parsed with the selected separator.") from e

    if filename.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(uploaded_file, header=None)
        except ValueError as e:
            raise ValueError("The uploaded spreadsheet could not be read.") from e

    raise ValueError("Unsupported file type")

def imageDataUri(path):
    image_path = Path(path)
    if not image_path.is_absolute():
        image_path = APP_DIR / image_path

    if not image_path.exists():
        return None

    suffix = image_path.suffix.lower()
    mime_type = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    elif suffix == ".svg":
        mime_type = "image/svg+xml"

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

# Verifies if the dataset contains a header row
def checkHeader(df):
    if len(df) < 1:
        return False

    totalVals = len(df.iloc[0])
    if totalVals == 0:
        return False

    stringVals = sum(isinstance(val, str) for val in df.iloc[0])

    if (stringVals / totalVals) < 0.85:
        return False
    else:
        return True

# Sets the first row as the header of the dataframe
def rowtoHeader(copyDf):
    df = copyDf.copy()
    fixedCol = [normalizeColumnName(val, index=i) for i, val in enumerate(df.iloc[0])]

    df.columns = makeUniqueColumns(fixedCol)
    df = df.drop(index=0)
    df = df.reset_index(drop=True)
    return df

# Applies the header row to the dataframe
def setHeader(copyDf):
    df = copyDf.copy()
    if (checkHeader(df)): 
        df = rowtoHeader(df)
    else:
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
    return df

def normalizeColumnName(value, index=0):
    if isinstance(value, str):
        cleaned = value.strip().lower()
        cleaned = cleaned.replace(" ", "_")
        return cleaned or f"col_{index}"

    if pd.isna(value):
        return f"col_{index}"

    return str(value).strip().lower() or f"col_{index}"

def makeUniqueColumns(columns):
    seen = {}
    unique_columns = []

    for column in columns:
        if column not in seen:
            seen[column] = 0
            unique_columns.append(column)
            continue

        seen[column] += 1
        unique_columns.append(f"{column}_{seen[column]}")

    return unique_columns

PLACEHOLDER_STRINGS = {'', 'na', 'n/a', 'nan', 'null', 'none', 'missing', '-', '--', '---'}
DATE_COLUMN_HINTS = ('date', 'time', 'timestamp', 'year', 'month', 'day')
MONTH_NAME_PATTERN = r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b'

# Checks whether a column name implies calendar/date values
def looksLikeDateColumn(column_name):
    if column_name is None:
        return False

    normalized = str(column_name).strip().lower()
    return any(hint in normalized for hint in DATE_COLUMN_HINTS)

# Normalizes values before date parsing
def normalizeDateText(series):
    text = series.astype("string").str.strip()
    missing_mask = text.str.lower().isin(PLACEHOLDER_STRINGS)
    return text.mask(missing_mask)

# Uses pandas' mixed parser when available, with a fallback for older pandas versions
def parseDatesWithPandas(values, dayfirst=False):
    try:
        converted = pd.to_datetime(values, format='mixed', dayfirst=dayfirst, errors='coerce')
    except (TypeError, ValueError):
        converted = pd.to_datetime(values, dayfirst=dayfirst, errors='coerce')

    return converted

# Parses compact numeric date representations without mistaking measurements for dates
def parseCompactDateValues(text, column_hint=False, dayfirst=False, force=False):
    cleaned = text.str.replace(r'\.0$', '', regex=True)
    non_null = cleaned.dropna()

    if non_null.empty:
        return None

    def try_format(mask, date_format, values=None):
        if mask.sum() / len(non_null) < 0.6:
            return None

        candidate_values = cleaned if values is None else values
        converted = pd.to_datetime(candidate_values.where(mask.reindex(cleaned.index, fill_value=False)), format=date_format, errors='coerce')
        success_rate = converted.notna().sum() / len(non_null)
        if success_rate >= 0.6 or force:
            return converted

        return None

    ymd_mask = non_null.str.match(r'^(?:18|19|20|21)\d{6}$', na=False)
    converted = try_format(ymd_mask, "%Y%m%d")
    if converted is not None:
        return converted

    ym_mask = non_null.str.match(r'^(?:18|19|20|21)\d{4}$', na=False)
    ym_values = cleaned + "01"
    converted = try_format(ym_mask, "%Y%m%d", values=ym_values)
    if converted is not None:
        return converted

    if dayfirst:
        local_format = "%d%m%Y"
    else:
        local_format = "%m%d%Y"

    local_mask = non_null.str.match(r'^\d{8}$', na=False)
    if column_hint or force:
        converted = try_format(local_mask, local_format)
        if converted is not None:
            return converted

    year_mask = non_null.str.match(r'^(?:18|19|20|21)\d{2}$', na=False)
    if column_hint or force:
        converted = try_format(year_mask, "%Y")
        if converted is not None:
            return converted

    numeric_values = pd.to_numeric(non_null, errors='coerce')
    numeric_rate = numeric_values.notna().sum() / len(non_null)
    if (column_hint or force) and numeric_rate >= 0.9:
        plausible_serial = numeric_values.between(20000, 60000).sum() / len(non_null)
        if plausible_serial >= 0.8:
            converted_non_null = pd.to_datetime(numeric_values, unit='D', origin='1899-12-30', errors='coerce')
            converted = pd.Series(pd.NaT, index=text.index, dtype='datetime64[ns]')
            converted.loc[converted_non_null.index] = converted_non_null
            return converted

    return None

# Estimates whether text values contain date separators, month names, or compact calendar values
def dateLikeTextRatio(text):
    non_null = text.dropna()

    if non_null.empty:
        return 0

    has_date_tokens = (
        non_null.str.contains(r'[/:\-T]', regex=True, na=False)
        | non_null.str.contains(MONTH_NAME_PATTERN, case=False, regex=True, na=False)
        | non_null.str.match(r'^(?:18|19|20|21)\d{2}(?:\d{2}){0,2}$', na=False)
    )
    return has_date_tokens.sum() / len(non_null)

# Parses and converts a column into datetime format
def dateTimeColumn(series, dayfirst=False, column_name=None, min_success=0.6, force=False):
    s = series.copy()

    if pd.api.types.is_datetime64_any_dtype(s):
        return s

    text = normalizeDateText(s)
    non_null_count = text.notna().sum()
    if non_null_count == 0:
        return series

    column_hint = looksLikeDateColumn(column_name)

    compact_converted = parseCompactDateValues(text, column_hint=column_hint, dayfirst=dayfirst, force=force)
    if compact_converted is not None:
        success_rate = compact_converted.notna().sum() / non_null_count
        if force or success_rate >= min_success:
            return compact_converted

    if not force and dateLikeTextRatio(text) < 0.5 and not column_hint:
        return series

    converted = parseDatesWithPandas(text, dayfirst=dayfirst)
    success_rate = converted.notna().sum() / non_null_count

    if force or success_rate >= min_success:
        return converted
        
    return series

# Repairs rows where leading metadata columns are missing before the date value
def repairRowsMissingLeadingColumns(copyDf, date_dayfirst=False, preferred_date_columns=None):
    df = copyDf.copy()
    preferred_date_columns = preferred_date_columns or []
    date_cols = [col for col in preferred_date_columns if col in df.columns]

    if not date_cols:
        date_cols = [col for col in df.columns if str(col).strip().lower() == 'date']

    if not date_cols:
        return df

    date_col = date_cols[0]
    date_index = df.columns.get_loc(date_col)

    if date_index <= 0:
        return df

    first_col = df.columns[0]
    first_as_date = dateTimeColumn(
        df[first_col],
        dayfirst=date_dayfirst,
        column_name=date_col,
        force=True
    )

    if not pd.api.types.is_datetime64_any_dtype(first_as_date):
        return df

    missing_before_date = df.iloc[:, 1:date_index + 1].isna().all(axis=1)
    shifted_rows = first_as_date.notna() & missing_before_date

    if not shifted_rows.any():
        return df

    values = df.to_numpy(dtype=object, copy=True)
    row_positions = np.flatnonzero(shifted_rows.to_numpy())

    for row_position in row_positions:
        original = values[row_position].copy()
        values[row_position, :] = np.nan
        values[row_position, date_index:] = original[:len(df.columns) - date_index]

    return pd.DataFrame(values, columns=df.columns)

# Identifies and converts date columns within the dataframe
def getDateTime(copyDf, date_dayfirst=False, preferred_date_columns=None):
    df = copyDf.copy()
    dateColumns = []
    preferred_date_columns = preferred_date_columns or []

    for col in preferred_date_columns:
        if col in df.columns:
            df[col] = dateTimeColumn(df[col], dayfirst=date_dayfirst, column_name=col, force=True)
            dateColumns.append(col)

    for col in df.columns:
        if col in dateColumns:
            continue
            
        non_null = df[col].dropna()
        if non_null.empty or len(non_null) < 3:
            continue

        sample = non_null.head(50)
        sample_converted = dateTimeColumn(
            sample,
            dayfirst=date_dayfirst,
            column_name=col,
            min_success=0.5
        )

        if pd.api.types.is_datetime64_any_dtype(sample_converted):
            converted = dateTimeColumn(df[col], dayfirst=date_dayfirst, column_name=col)
            if pd.api.types.is_datetime64_any_dtype(converted):
                df[col] = converted
                dateColumns.append(col)
        
    return df

# Computes grouped means of numeric columns for each date column
def displayDates(df):
    numericColumns = df.select_dtypes(include=['number']).columns
    output = {}

    dateCols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    if not dateCols:
        return None

    for col in dateCols:
        output[col] = df.groupby(col)[numericColumns].mean()

    return output

# Converts datetimes to decimal years so sub-annual data keeps its time spacing
def datetimeToDecimalYear(series):
    dates = pd.to_datetime(series, errors='coerce')
    decimal_years = pd.Series(np.nan, index=series.index, dtype=float)
    valid_dates = dates[dates.notna()]

    if valid_dates.empty:
        return decimal_years

    year_start = pd.to_datetime(valid_dates.dt.year.astype(str) + "-01-01")
    next_year_start = pd.to_datetime((valid_dates.dt.year + 1).astype(str) + "-01-01")
    elapsed = (valid_dates - year_start).dt.total_seconds()
    year_length = (next_year_start - year_start).dt.total_seconds()

    decimal_years.loc[valid_dates.index] = valid_dates.dt.year + (elapsed / year_length)
    return decimal_years

# Prepares the two selected columns for the embedded waveletAnalysis function
def prepareWaveletInput(df, time_col, value_col):
    analysis_df = df[[time_col, value_col]].copy()
    analysis_df["wavelet_time"] = datetimeToDecimalYear(analysis_df[time_col])
    analysis_df["wavelet_value"] = pd.to_numeric(analysis_df[value_col], errors='coerce')
    analysis_df = analysis_df.dropna(subset=["wavelet_time", "wavelet_value"])
    analysis_df = analysis_df.sort_values("wavelet_time")

    return analysis_df.groupby("wavelet_time", as_index=False)["wavelet_value"].mean()

def safeFilename(value, default="file", strip_extension=False):
    raw = str(value or "").replace("\\", "/")
    name = Path(raw).name
    if strip_extension:
        name = Path(name).stem

    cleaned = "".join(char if char.isalnum() else "_" for char in name.lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or default

def downloadFilename(value, extension, default="download"):
    extension = extension.lstrip(".")
    stem = safeFilename(value, default=default, strip_extension=True)
    return f"{stem}.{extension}"

def figureToPngBytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

def closeFigure(fig):
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError:
        return

# Performs Continuous Wavelet Transform inside the DataReader app
def waveletAnalysis(years, values, x_label="Year", y_label="Data", dj=0.2, w0=6.0, s0_mult=2.0, max_period=24, title=None):
    try:
        import matplotlib.pyplot as plt
        import pycwt as wavelet
    except ImportError as e:
        raise ImportError("Wavelet analysis requires matplotlib and pycwt to be installed.") from e

    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(years) < 2 or len(values) < 2:
        raise ValueError("Not enough data points for analysis.")

    dt = np.mean(np.diff(years))
    if np.isnan(dt) or dt <= 0:
        dt = 1

    mother = wavelet.Morlet(w0)

    std_dev = np.std(values)
    if std_dev <= 0:
        raise ValueError("Wavelet analysis requires variation in the selected data column.")

    normalized_values = (values - np.mean(values)) / std_dev

    if len(normalized_values) < 2:
        raise ValueError("Not enough data points for autocorrelation.")

    alpha_lag1 = np.corrcoef(normalized_values[:-1], normalized_values[1:])[0, 1]
    if np.isnan(alpha_lag1):
        alpha_lag1 = 0.0

    s0 = s0_mult * dt
    if s0 <= 0 or max_period <= s0:
        raise ValueError("Max period must be greater than the smallest wavelet scale.")

    J = max(1, int(np.log2(max_period / s0) / dj))

    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
        normalized_values, dt, dj=dj, s0=s0, J=J, wavelet=mother
    )

    signif, _ = wavelet.significance(1.0, dt, scales, 0, alpha=alpha_lag1, wavelet=mother)
    power = (np.abs(wave)) ** 2
    max_power = float(np.nanmax(power))
    if not np.isfinite(max_power) or max_power <= 0:
        raise ValueError("Wavelet analysis could not calculate positive power for the selected data.")

    sig95 = power / np.outer(signif, np.ones(len(normalized_values)))

    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(12, 6))

        contour = ax.contourf(years, np.log2(scales), power, levels=np.linspace(0, max_power, 100), cmap='jet')
        fig.colorbar(contour, ax=ax, label='Wavelet Power')

        try:
            ax.contour(years, np.log2(scales), sig95, [1], colors='black', linewidths=0.75)
        except Exception:
            pass

        ax.fill_between(years, np.log2(coi), np.log2(scales[-1]), color='gray', alpha=0.3, label='COI')
        ax.set_ylabel('Period (log2 scale)')
        ax.set_xlabel(x_label)

        tick_vals = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        tick_vals = [t for t in tick_vals if t <= max_period]
        if max_period not in tick_vals:
            tick_vals.append(max_period)
        ax.set_yticks(np.log2(tick_vals))
        ax.set_yticklabels([str(int(t)) for t in tick_vals])

        ax.set_title(title or f'Wavelet Power Spectrum (CWT) of {y_label}')
        ax.set_ylim(np.log2([scales[0], scales[-1]]))
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    return fig

def renderWaveletSection(workingDf):
    st.subheader("Wavelet Analysis")

    numeric_cols = workingDf.select_dtypes(include=np.number).columns.tolist()
    date_cols = [c for c in workingDf.columns if pd.api.types.is_datetime64_any_dtype(workingDf[c])]

    if not numeric_cols:
        st.warning("Wavelet analysis requires at least one numeric data column.")
        return

    if not date_cols:
        st.warning("Wavelet analysis requires at least one datetime column for the X-axis. Please convert a column to date in the Datatypes menu.")
        return

    col_time, col_value = st.columns(2)
    with col_time:
        time_col = st.selectbox(
            "Time column",
            date_cols,
            key="wavelet_time_col"
        )
    with col_value:
        value_col = st.selectbox(
            "Data column",
            numeric_cols,
            key="wavelet_value_col"
        )

    default_title = f"Wavelet Power Spectrum (CWT) of {value_col}"
    graph_title = st.text_input(
        "Graph title",
        value=default_title,
        key=f"wavelet_title_{time_col}_{value_col}"
    )

    with st.expander("Advanced settings", expanded=False):
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            dj_param = st.number_input("Scale resolution (dj)", min_value=0.05, max_value=1.0, value=0.2, step=0.05)
        with col_p2:
            w0_param = st.number_input("Morlet parameter (w0)", min_value=2.0, max_value=20.0, value=6.0, step=0.5)
        with col_p3:
            s0_param = st.number_input("Smallest scale multiplier", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        with col_p4:
            max_period_param = st.number_input("Max period (years)", min_value=2, max_value=1000, value=24, step=1)

    current_config = (
        time_col,
        value_col,
        graph_title,
        float(dj_param),
        float(w0_param),
        float(s0_param),
        int(max_period_param)
    )

    generated_now = False
    if st.button("Generate Wavelet Graph"):
        try:
            analysis_df = prepareWaveletInput(workingDf, time_col, value_col)

            if len(analysis_df) < 3:
                st.error("Not enough valid data in the selected columns to perform analysis.")
                return

            years = analysis_df["wavelet_time"].to_numpy(dtype=float)
            values = analysis_df["wavelet_value"].to_numpy(dtype=float)

            with st.spinner("Generating wavelet power spectrum..."):
                fig = waveletAnalysis(
                    years,
                    values,
                    x_label=str(time_col),
                    y_label=str(value_col),
                    dj=dj_param,
                    w0=w0_param,
                    s0_mult=s0_param,
                    max_period=max_period_param,
                    title=graph_title
                )

                try:
                    png_bytes = figureToPngBytes(fig)
                    st.pyplot(fig)
                finally:
                    closeFigure(fig)

                st.session_state[STATE_WAVELET_PNG] = png_bytes
                st.session_state[STATE_WAVELET_CONFIG] = current_config
                st.session_state[STATE_WAVELET_FILENAME] = downloadFilename(
                    f"wavelet_{safeFilename(value_col)}",
                    "png",
                    default="wavelet"
                )
                generated_now = True

        except Exception as e:
            logger.exception("Wavelet analysis failed")
            st.error("Wavelet analysis could not be completed for the selected columns and settings.")

    if (
        st.session_state.get(STATE_WAVELET_PNG) is not None
        and st.session_state.get(STATE_WAVELET_CONFIG) == current_config
    ):
        if not generated_now:
            st.image(st.session_state[STATE_WAVELET_PNG])

        download_name = st.text_input(
            "PNG filename",
            value=st.session_state.get(STATE_WAVELET_FILENAME, "wavelet.png"),
            max_chars=80,
            key="wavelet_download_filename"
        )
        download_name = downloadFilename(download_name, "png", default="wavelet")

        st.download_button(
            label="Download Graph as PNG",
            data=st.session_state[STATE_WAVELET_PNG],
            file_name=download_name,
            mime="image/png"
        )

# Cleans data by parsing values through the selected preset pipeline
@st.cache_data
def cleanData(copyDf, preset_name=NOAA_PRESET, date_dayfirst=False):
    df = copyDf.copy()
    preset_config = get_preset_config(preset_name)

    df = setHeader(df)
    df = normalize_missing_values(df)

    if preset_config["repair_missing_leading_dates"]:
        df = repairRowsMissingLeadingColumns(
            df,
            date_dayfirst=date_dayfirst,
            preferred_date_columns=preset_config["preferred_date_columns"]
        )

    df = getDateTime(
        df,
        date_dayfirst=date_dayfirst,
        preferred_date_columns=preset_config["preferred_date_columns"]
    )
    df = inferNumericColumns(df)
    df = applyNumericPlaceholders(df, preset_config)

    return df

def normalize_missing_values(copyDf):
    df = copyDf.copy()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        mask = df[col].str.lower().isin(PLACEHOLDER_STRINGS)
        df[col] = df[col].where(~mask, other=np.nan)

    return df

def inferNumericColumns(copyDf):
    df = copyDf.copy()
    for col in list(df.select_dtypes(include="object").columns):
        non_null = df[col].dropna()
        if non_null.empty:
            continue

        converted = pd.to_numeric(non_null, errors='coerce')
        success_rate = converted.notna().sum() / len(non_null)

        if success_rate >= 0.75:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            continue

        has_thousands = non_null.str.contains(r'^\-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$', na=False)
        if has_thousands.sum() / len(non_null) >= 0.5:
            fixed = non_null.str.replace(',', '', regex=False)
            converted = pd.to_numeric(fixed, errors='coerce')
            success_rate = converted.notna().sum() / len(non_null)
            if success_rate >= 0.75:
                df[col] = df[col].str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                continue

        has_comma_decimal = non_null.str.contains(r'^\-?\d+,\d{1,2}$|^\-?\d+,\d{4,}$', na=False)
        if has_comma_decimal.sum() / len(non_null) >= 0.5:
            fixed = non_null.str.replace(',', '.', regex=False)
            converted = pd.to_numeric(fixed, errors='coerce')
            success_rate = converted.notna().sum() / len(non_null)
            if success_rate >= 0.75:
                df[col] = df[col].str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                continue

    return df

def applyNumericPlaceholders(copyDf, preset_config):
    df = copyDf.copy()
    placeholder_columns = preset_config.get("numeric_placeholder_columns", set())
    placeholders = preset_config.get("numeric_placeholders", set())

    if not placeholder_columns or not placeholders:
        return df

    replacement_map = {value: np.nan for value in placeholders}
    for col in df.select_dtypes(include=['number']).columns:
        if str(col).lower() in placeholder_columns:
            df[col] = df[col].replace(replacement_map)

    return df

# Gathers unique values in each dataframe column
@st.cache_data
def displayUniques(df):
    uniques = {}
    for column in df.columns:
        uniques[column] = df[column].unique()
    return uniques

# Filters a dataframe based on the provided percentile thresholds
@st.cache_data
def findPercentiles(df, dfPercent, choice, selected_col=None):
    if selected_col is None or selected_col not in df.columns:
        return pd.DataFrame(), None

    threshold = dfPercent[selected_col]

    if choice in ("Under percentile", "At or below percentile"):
        mask = df[selected_col] <= threshold
    elif choice in ("Over percentile", "At or above percentile"):
        mask = df[selected_col] >= threshold
    else:
        return pd.DataFrame(), threshold

    filtered = df.loc[mask].copy()
    return filtered, threshold

# Retrieves statistical properties for the numerical columns
def showMathInfo(df):
    numericCols = df.select_dtypes(include=['number'])
    if numericCols.shape[1] == 0:
        return None

    dfStats = numericCols.describe()
    return dfStats
        
# Retrieves statistical properties for categorical columns
def showCategoricalInfo(df):
    catCols = df.select_dtypes(include=['object', 'category', 'string'])
    if catCols.shape[1] == 0:
        return None

    return catCols.describe()

# Allows the user to download a generated subset of statistics as a CSV
def saveStatistics(df, statistic, rotate=True, widget_prefix="save"):
    if df is not None and not df.empty:
        if rotate:
            df = df.T

        if len(df.columns) > 0:
            colToSave = st.selectbox("Select column to save", df.columns.tolist(), key=f"{widget_prefix}_col_select")

            defaultFilename = f"{st.session_state.get(STATE_FILENAME, 'data')}{statistic}_{colToSave}"
            fileName = st.text_input("Filename for download", defaultFilename, max_chars=40, key=f"{widget_prefix}_filename")
            download_name = downloadFilename(fileName, "csv", default="statistics")

            csv = df.to_csv(index=True).encode('utf-8')
            st.download_button(
                label=f"Download {download_name}",
                data=csv,
                file_name=download_name,
                mime="text/csv",
                key=f"{widget_prefix}_download"
            )

def view_head(df, num_rows=10):
    return df.head(num_rows)

def rename_column(df, old_name, new_name):
    if old_name not in df.columns:
        raise KeyError(f"Column {old_name} not found")

    new_name = str(new_name).strip()
    if not new_name:
        raise ValueError("New column name cannot be empty")

    if new_name != old_name and new_name in df.columns:
        raise ValueError(f"Column {new_name} already exists")

    return df.rename(columns={old_name: new_name})

def delete_columns(df, columns):
    if not columns:
        raise ValueError("Select at least one column to delete")

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {', '.join(missing)}")

    return df.drop(columns=columns)

def rows_matching_condition(df, column, condition, value):
    if column not in df.columns:
        raise KeyError(f"Column {column} not found")

    if condition == "Greater than":
        return df[df[column] > value].index.tolist()

    if condition == "Less than":
        return df[df[column] < value].index.tolist()

    if condition == "Equal to":
        return df[np.isclose(df[column], value, equal_nan=False)].index.tolist()

    raise ValueError(f"Unknown condition {condition}")

def delete_rows(df, rows):
    missing = [row for row in rows if row not in df.index]
    if missing:
        raise KeyError(f"Rows not found: {missing}")

    return df.drop(index=rows).reset_index(drop=True)

def drop_duplicate_rows(df, subset_cols=None, keep_option="first"):
    keep_value = False if keep_option == "none" else keep_option
    subset_cols = subset_cols or None
    return df.drop_duplicates(subset=subset_cols, keep=keep_value)

def sort_dataframe(df, column, ascending=True):
    if column not in df.columns:
        raise KeyError(f"Column {column} not found")

    return df.sort_values(by=column, ascending=ascending).reset_index(drop=True)

def convert_column_type(df, column, dtype_choice, dayfirst=False):
    if column not in df.columns:
        raise KeyError(f"Column {column} not found")

    dfLocal = df.copy()

    if dtype_choice == 'int':
        dfLocal[column] = pd.to_numeric(dfLocal[column], errors='coerce').astype('Int64')
    elif dtype_choice == 'float':
        dfLocal[column] = pd.to_numeric(dfLocal[column], errors='coerce').astype(float)
    elif dtype_choice == 'string':
        dfLocal[column] = dfLocal[column].astype("string")
    elif dtype_choice == 'date':
        dfLocal[column] = dateTimeColumn(
            dfLocal[column],
            dayfirst=dayfirst,
            column_name=column,
            force=True
        )
    else:
        raise ValueError(f"Unknown datatype {dtype_choice}")

    return dfLocal

def extract_date_components(df, column, components):
    if column not in df.columns:
        raise KeyError(f"Column {column} not found")

    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        raise TypeError(f"Column {column} is not datetime type")

    dfLocal = df.copy()
    component_map = {
        'year': dfLocal[column].dt.year,
        'month': dfLocal[column].dt.month,
        'day': dfLocal[column].dt.day,
        'hour': dfLocal[column].dt.hour,
        'minute': dfLocal[column].dt.minute,
        'second': dfLocal[column].dt.second,
    }

    for component in components:
        if component not in component_map:
            raise ValueError(f"Unknown component {component}")
        dfLocal[f"{column}_{component}"] = component_map[component]

    return dfLocal

if __name__ == "__main__":
    main()
