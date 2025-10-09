import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def main():
    # Streamlit UI and mapping all functions
    st.set_page_config(page_title="DataReader Web", layout="wide", page_icon="images/RamapoArch.png")

    # Header Bar
    col1, col2 = st.columns([1, 4])

    with col1:
        st.image("images/whiteRamapoLogo.png", width=200)

    with col2:
        st.markdown("""
        <h2 style='text-align:left; color:white;'>DataReader Web</h2>
        <p style='text-align:left; color:white;'>Created by Aiden Cabrera for the Ramapo Climate Research Group</p>

        """, unsafe_allow_html=True)

    # Ramapo Theme
    st.markdown("""
        <style>
        [data-testid="stHeader"] {display: none;}  /* Hide the default black bar */

       /* Custom Header */
        .custom-header {
            background-color: #862633;  /* Ramapo maroon */
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.5rem 1rem;
            border-bottom: 3px solid #C41E1E; /* Ramapo red accent */
            position: sticky;
            top: 0;
            z-index: 999;
            box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.3);
        }

        /* File uploader outer container */
        [data-testid="stFileUploader"] {
            background-color: #862633;  /* maroon outer layer */
            max-width: 1200px;
            gap: 2rem;
            padding: .5rem;
            border-radius: 10px;
            border: 2px solid #C41E1E;  /* red border accent */
            display: flex;
            justify-content: center;
        }

        /* File uploader inner box (black center) */
        [data-testid="stFileUploader"] > div {
            display: flex; 
            background-color: #000000 !important;  /* black center */
            border-radius: 8px;
            padding: 1rem !important;
            gap: 10rem;
            width: 100%;
            color: white !important;
            align-items: center;

        /* Selectboxes in header */
        div[data-baseweb="select"] > div {
            background-color: #C41E1E !important;
            color: white !important;
            border: none !important;
            border-radius: 6px;
            padding: 2px 8px;
        }
        div[data-baseweb="select"] svg {
            fill: white !important;
        }

        /* --- Main content area --- */
        [data-testid="stAppViewContainer"] {
            background-color: #25282A;   /* dark background */
            color: #FFFFFF;
        }

        /* --- Buttons --- */
        .stButton > button {
            background-color: #C41E1E;   /* red */
            color: #FFFFFF;
            border: none;
            border-radius: 6px;
            padding: 0.5em 1em;
            font-weight: 600;
        }
        .stButton > button:hover {
            background-color: #A42228;   /* darker red on hover */
            transform: scale(1.02);
            transition: all 0.2s ease-in-out;
        }

        /* --- Dataframes --- */
        [data-testid="stDataFrame"] {
            border: 1px solid #313436;
            border-radius: 6px;
            overflow: hidden;
        }

        /* Remove Streamlit’s default element toolbar */
        [data-testid="stElementToolbar"] {
            display: none !important;
        }
        </style>
        """, unsafe_allow_html=True)


    # File uploader
    uploadedFile = st.file_uploader("Upload CSV, Excel, or TXT file", type=["csv", "xlsx", "xls", "txt"])

    # Main site loop
    if uploadedFile is not None:

        dfRaw = st.session_state.get('dfRaw')

        # Loads file, cleans data, and only reloads during new upload
        if st.session_state.get('workingDf') is None:
            try:
                if uploadedFile.name.endswith((".csv", ".txt")):
                    dfRaw = pd.read_csv(uploadedFile, header=None) 
                elif uploadedFile.name.endswith((".xlsx", ".xls")):
                    dfRaw = pd.read_excel(uploadedFile)
                else:
                    raise ValueError("Unsupported file type")

                st.success("File uploaded successfully.")
                
                # Store original filename
                st.session_state.filename = uploadedFile.name.rsplit(".", 1)[0]
                st.session_state.dfRaw = dfRaw.copy()
                
                # Clean only once
                workingDf = cleanData(dfRaw)
                st.session_state.workingDf = workingDf
        
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.session_state.workingDf = None
            # if workingDf is not None and st.button("Rotate data?"):
            #     try:
            #         dfRaw = st.session_state.dfRaw

            #         # --- Rotate safely ---
            #         rotated = dfRaw.T.copy().reset_index(drop=True)

            #         # Use the first transposed row as new column headers
            #         rotated.columns = rotated.iloc[0].astype(str)
            #         rotated = rotated.drop(rotated.index[0]).reset_index(drop=True)

            #         # Clean the rotated data
            #         st.session_state.workingDf = cleanData(rotated)

            #         # Success message with placeholder (can be cleared)
            #         msg_placeholder = st.empty()
            #         msg_placeholder.success("Data rotated, headers checked, and cleaned successfully.")

            #         # Show new rotated + cleaned data
            #         st.dataframe(st.session_state.workingDf.head())

            #     except Exception as e:
            #         st.error(f"Error rotating data: {e}")
            
            # Preview
        
        workingDf = st.session_state.get('workingDf')

        with st.expander("Preview data"):
            st.dataframe(workingDf.head(10))

        mainMenu = st.selectbox(
            "Select an option to analyze or edit data",
            [
                "Statistics",
                "Datatypes",
                "Reset Data",
                "Edit Data",
                "Visualize Data",
                "Save / Export"
            ],
            key="mainMenu",
            label_visibility="collapsed",
            width=500
        )

        # Numeric Statistics 
        if mainMenu == "Statistics":
            catCols = workingDf.select_dtypes(exclude=['object']).columns.tolist()
            numericCols = workingDf.select_dtypes(include=['number']).columns.tolist()
            statOption = st.selectbox("Choose statistics type", ["Numeric", "Categorical"])
            if numericCols and statOption == "Numeric":
                try:
                    st.subheader("Numeric Statistics")
                    dfStats = showMathInfo(workingDf)

                    statOption = st.radio(
                        "Choose a numeric statistic to view",
                        ["Count per column", "Mean", "Std", "Min/Max rows", "Percentile", "All statistics"]
                    )

                    # Count per column
                    if statOption == "Count per column":
                        colChoice = st.selectbox("Select column or choose All", ["All"] + numericCols)
                        if colChoice == "All":
                            st.write("Count of non-null values per column:")
                            st.write(dfStats.loc['count'])
                            saveStatistics(dfStats.loc[['count']], "_countPerColumn")
                        else:
                            st.write(f"Count for column '{colChoice}': {dfStats.at['count', colChoice]}")
                            saveStatistics(pd.DataFrame({colChoice: [dfStats.at['count', colChoice]]}), f"_{colChoice}_count")

                    # Mean
                    elif statOption == "Mean":
                        colChoice = st.selectbox("Select column or choose All", ["All"] + numericCols)
                        if colChoice == "All":
                            st.write("Mean per column:")
                            st.write(dfStats.loc['mean'])
                            saveStatistics(dfStats.loc[['mean']], "_meanPerColumn")
                        else:
                            st.write(f"Mean for column '{colChoice}': {dfStats.at['mean', colChoice]}")
                            saveStatistics(pd.DataFrame({colChoice: [dfStats.at['mean', colChoice]]}), f"_{colChoice}_mean")

                    # Standard Deviation
                    elif statOption == "Std":
                        colChoice = st.selectbox("Select column or choose All", ["All"] + numericCols)
                        if colChoice == "All":
                            st.write("Standard deviation per column:")
                            st.write(dfStats.loc['std'])
                            saveStatistics(dfStats.loc[['std']], "_stdPerColumn")
                        else:
                            st.write(f"Standard deviation for column '{colChoice}': {dfStats.at['std', colChoice]}")
                            saveStatistics(dfStats.loc[['std']], "_stdPerColumn")
                    # Min/Max Rows 
                    elif statOption == "Min/Max rows":
                        colChoice = st.selectbox("Select column for min/max or choose All", ["All"] + numericCols)
                        def minMaxRow(column):
                            minVal = dfStats.at['min', column]
                            maxVal = dfStats.at['max', column]
                            minRow = workingDf[workingDf[column] == minVal]
                            maxRow = workingDf[workingDf[column] == maxVal]
                            st.write(f"Column: {column}")
                            if not minRow.empty:
                                st.write("Min row(s):")
                                st.dataframe(minRow)
                                saveStatistics(minRow, f"_{column}_minRow")
                            else:
                                st.write("No min row found")
                            if not maxRow.empty:
                                st.write("Max row(s):")
                                st.dataframe(maxRow)
                                saveStatistics(maxRow, f"_{column}_maxRow")
                            else:
                                st.write("No max row found")
                        if colChoice == "All":
                            for col in numericCols:
                                minMaxRow(col)
                        else:
                            minMaxRow(colChoice)
                    # Percentiles 
                    elif statOption == "Percentile":
                        userPercent = st.number_input(
                            "Percentile (0-100)", min_value=0.0, max_value=100.0, value=50.0, step=0.1
                        )

                        if st.button("Show Percentile"):
                            percent = round(userPercent / 100.0, 4)
                            dfPercent = workingDf.quantile(percent, numeric_only=True)

                            # Save to session state
                            st.session_state.dfPercent = dfPercent
                            st.session_state.userPercent = userPercent

                            st.write(f"{userPercent}% percentile values")
                            st.dataframe(dfPercent)

                        # Only show radio if dfPercent exists
                        if "dfPercent" in st.session_state:
                            choice = st.radio(
                                "Show values relative to percentile:",
                                ("None", "Under percentile", "Over percentile"),
                                key="percentile_choice"
                            )

                            if choice != "None":
                                results = findPercentiles(
                                    workingDf, st.session_state.dfPercent, choice
                                )
                                st.write(f"Number of values {choice.lower()} for each column:")
                                st.dataframe(results)
                                saveStatistics(results, f"_values_{choice.split()[0].lower()}Percentile")
                    # All Statistics
                    elif statOption == "All statistics":
                        st.dataframe(dfStats)
                        saveStatistics(dfStats, "_numericStatistics", rotate=False)
                except Exception as e:
                    st.error(str(e))
        
            if catCols and statOption == "Categorical":
                    try:
                        st.subheader("Categorical Statistics")
                        catStats = showCategoricalInfo(workingDf)

                        statOption = st.radio(
                            "Choose a categorical statistic to view",
                            ["Unique counts", "Most frequent values", "Least frequent values", "All statistics"]
                        )

                        # Unique counts
                        if statOption == "Unique counts":
                            uniqueCounts = workingDf[catCols].nunique()
                            st.write("Number of unique values per column:")
                            st.dataframe(uniqueCounts)
                            saveStatistics(uniqueCounts, "_uniqueCounts")

                        # Most frequent values
                        elif statOption == "Most frequent values":
                            colChoice = st.selectbox("Select column to view most frequent values", catCols)
                            topFreq = workingDf[colChoice].value_counts().head(10)
                            st.write(f"Top 10 most frequent values for '{colChoice}':")
                            st.dataframe(topFreq)
                            saveStatistics(topFreq, f"_{colChoice}_mostFrequent")

                        # Least frequent values
                        elif statOption == "Least frequent values":
                            colChoice = st.selectbox("Select column to view least frequent values", catCols)
                            lowFreq = workingDf[colChoice].value_counts().tail(10)
                            st.write(f"10 least frequent values for '{colChoice}':")
                            st.dataframe(lowFreq)
                            saveStatistics(lowFreq, f"_{colChoice}_leastFrequent")

                        # All statistics (summary view)
                        elif statOption == "All statistics":
                            st.write("Full categorical statistics:")
                            st.dataframe(catStats)
                            saveStatistics(catStats, "_categoricalStatistics", rotate=False)

                    except Exception as e:
                        st.error(str(e))
        # Datatypes
        elif mainMenu == "Datatypes":
            st.subheader("Column datatypes")
            st.write(workingDf.dtypes)

            editAction = st.selectbox("Edit datatypes", ["Change datatype"])
            if editAction == "None":
                pass
            elif editAction == "Change datatype":
                colToChange = st.selectbox("Select column to change datatype", workingDf.columns.tolist())
                dtypeChoice = st.selectbox("Target datatype", ["int", "float", "string", "date"])
                if st.button("Change datatype"):
                    try:
                        workingDf = editData(workingDf, 'changeDatatype', colToChange=colToChange, dtypeChoice=dtypeChoice)
                        st.session_state.workingDf = workingDf
                        st.success(f"Converted {colToChange} to {dtypeChoice}")
                    except Exception as e:
                        st.error(str(e))

        # Edit Data 
        elif mainMenu == "Edit Data":
            st.subheader("Edit data")
            editAction = st.selectbox("Choose edit action", ["View data", "Rename column", "Delete column", "Drop duplicates", "Sort by column", "Change Date"])
            if editAction == "View data":
                numShow = st.number_input("Rows to show (min. 10)", min_value=10, max_value=workingDf.shape[0], value=10)
                st.dataframe(editData(workingDf, 'viewHead', numShow=numShow))

            elif editAction == "Rename column":
                colToEdit = st.selectbox("Select column to rename", workingDf.columns.tolist())
                newColName = st.text_input("New column")
                if st.button("Rename column"):
                    try:
                        workingDf = editData(workingDf, 'renameColumn', colToEdit=colToEdit, newColName=newColName)
                        st.session_state.workingDf = workingDf
                        st.success(f"Renamed {colToEdit} to {newColName}")
                        st.dataframe(workingDf.head())
                    except Exception as e:
                        st.error(str(e))

            elif editAction == "Delete column":
                colsToDelete = []
                for col in workingDf.columns:
                    if st.checkbox(f"Delete {col}"):
                        colsToDelete.append(col)

                if st.button("Delete selected columns"):
                    try:
                        for col in colsToDelete:
                            workingDf = editData(workingDf, 'deleteColumn', colToDelete=col)
                        st.session_state.workingDf = workingDf
                        st.success(f"Deleted columns: {', '.join(colsToDelete)}")
                        st.dataframe(workingDf.head())
                    except Exception as e:
                        st.error(str(e))

            elif editAction == "Drop duplicates":
                subsetCols = st.multiselect("Subset columns for duplicate detection (empty = all columns)", workingDf.columns.tolist())
                keepOption = st.selectbox("Keep which", ["first", "last", "none"])
                if st.button("Drop duplicates"):
                    before = len(workingDf)
                    if subsetCols:
                        if keepOption == "none":
                            workingDf = workingDf.drop_duplicates(subset=subsetCols, keep=False)
                        else:
                            workingDf = workingDf.drop_duplicates(subset=subsetCols, keep=keepOption)
                    else:
                        if keepOption == "none":
                            workingDf = workingDf.drop_duplicates(keep=False)
                        else:
                            workingDf = workingDf.drop_duplicates(keep=keepOption)
                    st.session_state.workingDf = workingDf
                    after = len(workingDf)
                    st.success(f"Dropped duplicates, rows before: {before}, rows after: {after}")

            elif editAction == "Sort by column":
                sortCol = st.selectbox("Select column to sort by", workingDf.columns.tolist())
                ascending = st.checkbox("Sort ascending", value=True)
                if st.button("Sort"):
                    workingDf = workingDf.sort_values(by=sortCol, ascending=ascending).reset_index(drop=True)
                    st.session_state.workingDf = workingDf
                    st.success(f"Sorted by {sortCol} {'ascending' if ascending else 'descending'}")
                    st.dataframe(workingDf.head())
            
            elif editAction == "Change Date":
                dateCols = [col for col in workingDf.columns if pd.api.types.is_datetime64_any_dtype(workingDf[col])]
                if dateCols:
                    colToEdit = st.selectbox("Select datetime column", dateCols)
                    # Checkboxes for components
                    components = []
                    if st.checkbox("Year"): components.append('year')
                    if st.checkbox("Month"): components.append('month')
                    if st.checkbox("Day"): components.append('day')
                    if st.checkbox("Hour"): components.append('hour')
                    if st.checkbox("Minute"): components.append('minute')
                    if st.checkbox("Second"): components.append('second')

                    if st.button("Apply"):
                        workingDf = editData(workingDf, 'changeDate', colToEdit=colToEdit, components=components)
                        st.session_state.workingDf = workingDf
                        st.success("Datetime components extracted")
                        st.dataframe(workingDf.head())

            # Visualize Data
            elif mainMenu == "Visualize Data":
                selectVisual = st.select_box("Select visualization type", ["Display Dates", "Display Uniques", "Visualize data"])
                if selectVisual == "Display Dates":
                    st.subheader("Display date-grouped numeric means")
                    dateOutputs = displayDates(workingDf)
                    if not dateOutputs:
                        st.info("No date columns found")
                    else:
                        for k, v in dateOutputs.items():
                            st.write(f"Group by {k}")
                            st.dataframe(v)
                            saveStatistics(v, f"_dateGroupedBy_{k}")
                elif selectVisual == "Display Uniques":
                    st.subheader("Unique values per column")
                    uniques = displayUniques(workingDf)
                    for col, vals in uniques.items():
                        st.write(f"{col} ({len(vals)} unique)")
                        st.write(vals)

                elif selectVisual == "Visualize data":
                    st.subheader("Visualize data")

                    numericCols = workingDf.select_dtypes(include=['number']).columns.tolist()
                    categoricalCols = workingDf.select_dtypes(exclude=['number']).columns.tolist()
                    vizType = st.selectbox("Choose visualization", ["Histogram", "Boxplot", "Scatter", "Bar (categorical)", "Correlation heatmap"])

                    if vizType == "Histogram":
                        if not numericCols:
                            st.info("No numeric columns available")
                        else:
                            colChoice = st.selectbox("Select numeric column", numericCols)
                            numBins = st.slider("Bins", min_value=5, max_value=200, value=20)
                            fig = px.histogram(
                                workingDf,
                                x=colChoice,
                                nbins=numBins,
                                title=f"Histogram of {colChoice}",
                                color_discrete_sequence=["#C41E1E"]
                            )
                            fig.update_layout(
                                template="plotly_dark",
                                xaxis_title=colChoice,
                                yaxis_title="Frequency"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    elif vizType == "Boxplot":
                        if not numericCols:
                            st.info("No numeric columns available")
                        else:
                            colChoice = st.selectbox("Select numeric column", numericCols)
                            fig = px.box(
                                workingDf,
                                y=colChoice,
                                title=f"Boxplot of {colChoice}",
                                color_discrete_sequence=["#C41E1E"]
                            )
                            fig.update_layout(
                                template="plotly_dark",
                                yaxis_title=colChoice
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    elif vizType == "Scatter":
                        if len(numericCols) < 2:
                            st.info("Need at least two numeric columns for scatter plot")
                        else:
                            xCol = st.selectbox("X axis", numericCols)
                            yCol = st.selectbox("Y axis", [c for c in numericCols if c != xCol])
                            fig = px.scatter(
                                workingDf,
                                x=xCol,
                                y=yCol,
                                title=f"Scatter: {xCol} vs {yCol}",
                                color_discrete_sequence=["#C41E1E"]
                            )
                            fig.update_traces(marker=dict(size=8, opacity=0.7))
                            fig.update_layout(
                                template="plotly_dark",
                                xaxis_title=xCol,
                                yaxis_title=yCol
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    elif vizType == "Bar (categorical)":
                        if not categoricalCols:
                            st.info("No categorical columns")
                        else:
                            colChoice = st.selectbox("Select categorical column", categoricalCols)
                            valueCounts = workingDf[colChoice].value_counts().head(50)
                            fig = px.bar(
                                x=valueCounts.values,
                                y=valueCounts.index,
                                orientation="h",
                                title=f"Top 50 Categories in {colChoice}",
                                labels={"x": "Count", "y": colChoice},
                                color_discrete_sequence=["#C41E1E"]
                            )
                            fig.update_layout(
                                template="plotly_dark",
                                yaxis=dict(autorange="reversed")
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    elif vizType == "Correlation heatmap":
                        if len(numericCols) < 2:
                            st.info("Need at least two numeric columns")
                        else:
                            corrMatrix = workingDf[numericCols].corr()
                            fig = px.imshow(
                                corrMatrix,
                                text_auto=True,
                                color_continuous_scale="RdBu_r",
                                title="Correlation Heatmap",
                                aspect="auto"
                            )
                            fig.update_layout(template="plotly_dark")
                            st.plotly_chart(fig, use_container_width=True)


        # Reset Data
        elif mainMenu == "Reset Data":
            st.write("Use these actions to clean the working data. Changes are applied to the working copy.")

            if st.button("Reset working data to original upload"):
                workingDf = st.session_state.dfRaw
                st.success("Reset complete")
                st.dataframe(workingDf.head())

        # Save / Export 
        elif mainMenu == "Save / Export":
            st.subheader("Save or export working data")

            # Default = original + "_edited.csv"
            defaultFilename = "export.csv"
            if "dfRaw" in st.session_state and "filename" in st.session_state.filename:
                base = st.session_state.filename
                defaultFilename = f"{base}_edited.csv"

            # Text input so user can rename before downloading
            newFilename = st.text_input("Filename for download", defaultFilename, max_chars=20)
            st.button("Apply")

            # Convert working df to CSV
            csv = workingDf.to_csv(index=False).encode("utf-8")

            # Download button
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=newFilename,
                mime="text/csv"
            )

        # Update session state workingDf
        st.session_state.workingDf = workingDf

# Helper functions 
def checkHeader(copyDf):
    df = copyDf.copy()
    if len(df) < 1:
        return False

    totalVals = len(df.iloc[0])
    stringVals = sum(isinstance(val, str) for val in df.iloc[0])

    if (stringVals / totalVals) < 0.85:
        return False
    else:
        return True

def rowtoHeader(copyDf):
    df = copyDf.copy()
    # Convert first row values to lowercase if they are strings
    fixedCol = []
    for val in df.iloc[0]:
        if isinstance(val, str):
            fixedCol.append(val.strip().lower())
        else:
            fixedCol.append(val)

    df.columns = fixedCol
    df = df.drop(index=0)
    df = df.reset_index(drop=True)  # Remove the old header row, reset index
    return df

def setHeader(copyDf):
    df = copyDf.copy()
    # If header exists, convert first row into header
    if checkHeader(df) or (st.button("Click if a header exists?")): 
        df = rowtoHeader(df)
    else:
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
    return df

@st.cache_data
def dateTimeColumn(series):
    s = series.copy()

    dateFormats = ["%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%d-%m-%Y",
                   "%m/%d/%y", "%m-%d-%y", "%d/%m/%y", "%d-%m-%y"]

    timeFormats = ["%H:%M:%S", "%H:%M", "%H-%M-%S", "%H-%M",
                   "%H.%M.%S", "%H.%M"]

    genericFormats = [
        "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m-%d-%Y %H:%M:%S", "%m-%d-%Y %H:%M",
        "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
        "%m/%d/%y %H:%M:%S", "%m/%d/%y %H:%M", "%m-%d-%y %H:%M:%S", "%m-%d-%y %H:%M",
        "%d/%m/%y %H:%M:%S", "%d/%m/%y %H:%M", "%d-%m-%y %H:%M:%S", "%d-%m-%y %H:%M"
    ]

    successful = False
    # Try generic formats first
    for fmt in genericFormats + dateFormats + timeFormats:
        try:
            converted = pd.to_datetime(s, format=fmt, errors='coerce')
            # if conversion produced some dates (not all NaT), accept
            if not converted.isna().all():
                # When convert, keep original non-convertible values as NaT
                s = converted
                successful = True
                break
        except Exception:
            continue

    if not successful:
        return series
    return s

def getDateTime(copyDf):
    df = copyDf.copy()
    dateColumns = []
    
    # Use stored selection if available
    if 'dateCols' in st.session_state:
        for col in st.session_state.dateCols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                dateColumns.append(col)
    
    st.session_state.dateCols = dateColumns

    # Rename date columns
    if len(dateColumns) == 1:
        df = df.rename(columns={dateColumns[0]: 'date'})
    elif len(dateColumns) > 1:
        df = df.rename(columns={dateColumns[0]: 'date1'})
    return df

@st.cache_data
def displayDates(df):
    # Numeric columns to aggregate
    numericColumns = df.select_dtypes(include=['number']).columns
    output = {}

    # Collect all datetime columns
    dateCols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    if not dateCols:
        return None

    for col in dateCols:
        # Group by exact date
        output[col] = df.groupby(col)[numericColumns].mean()

    return output

@st.cache_data
def cleanData(copyDf):
    df = copyDf.copy()

    df = setHeader(df)
    # Remove commas from numbers and replacing with periods
    for column in df.select_dtypes(include="object").columns:  # Only doing operation on strings
        df[column] = df[column].str.replace(',', '.', regex=True) 
        # Check for "/", "-" 
        if df[column].str.contains(r'[\/\-]').any():
            df[column] = dateTimeColumn(df[column])

    # Convert strings to numbers
    # Do it FIRST because it might convert numbers to date incorrectly otherwise
    for column in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[column], errors='coerce')
        # Does NOT have NaN, convert to numeric
        if not (converted.isna().sum() / len(converted) > 0.85):
            df[column] = converted 


    # Convert strings to dates
    df = getDateTime(df)

    # Replace common placeholders with NaN
    placeholders = [-999, 999, -9, 9999, 'NA', 'NaN', 'null', 'None', '', 'missing', -200]
    for col in df.columns:
        for pch in placeholders:
            df[col] = df[col].replace(to_replace=pch, value=np.nan)

    # Drop rows/columns with >= 75% missing values 
    # df = df.dropna(axis=0, thresh=(len(df.columns) * .2))
    # df = df.dropna(axis=1, thresh=(len(df.columns) * .2))

    # Clean string columns: remove special characters, lowercase
    for column in df.select_dtypes(include="object").columns:
        if pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].str.replace(r'[^\w\s]', '', regex=True)
            df[column] = df[column].str.lower()

    return df

@st.cache_data
def displayUniques(df):
    uniques = {}
    for column in df.columns:
        uniques[column] = df[column].unique()
    return uniques

@st.cache_data
def findPercentiles(df, dfPercent, choice):
    results = {}
    numeric_df = df.select_dtypes(include=['number'])

    for col in dfPercent.index:
        if col not in numeric_df.columns:
            continue
        threshold = dfPercent[col]

        if choice == "Under percentile":
            values = numeric_df.loc[numeric_df[col] < threshold, col].tolist()
        elif choice == "Over percentile":
            values = numeric_df.loc[numeric_df[col] > threshold, col].tolist()
        else:
            values = []

        results[col] = {
            "Count": len(values),
            "Values": values
        }

    return pd.DataFrame(results).T

def showMathInfo(df):
    numericCols = df.select_dtypes(include=['number'])
    if numericCols.shape[1] == 0:
        st.warning("No numeric columns found in the dataset.")
        return None

    dfStats = numericCols.describe()
    return dfStats
        
def showCategoricalInfo(df):
    catCols = df.select_dtypes(include=['object', 'category'])
    if catCols.shape[1] == 0:
        st.warning("No categorical columns found in the dataset.")
        return None

    return catCols.describe()

def saveStatistics(df, statistic, rotate=True):
    if df is not None and not df.empty:
        if rotate:
            df = df.T
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {st.session_state.filename + statistic}.csv",
            data=csv,
            file_name=f"{st.session_state.filename + statistic}.csv",
            mime="text/csv"
        )

# Menu Functions
def editData(df, action, **kwargs):
    # action is a string indicating what user wants
    # kwargs vary by action
    dfLocal = df.copy()
    if action == 'viewHead':
        numShow = kwargs.get('numShow', 5)
        return dfLocal.head(numShow)
    elif action == 'renameColumn':
        colToEdit = kwargs.get('colToEdit')
        newColName = kwargs.get('newColName')
        if colToEdit is None or newColName is None:
            raise ValueError("renameColumn requires colToEdit and newColName")
        dfLocal = dfLocal.rename(columns={colToEdit: newColName})
        return dfLocal
    elif action == 'deleteColumn':
        colToDelete = kwargs.get('colToDelete')
        if colToDelete in dfLocal.columns:
            dfLocal = dfLocal.drop(columns=[colToDelete])
            return dfLocal
        else:
            raise KeyError(f"Column {colToDelete} not found")
    elif action == 'changeDatatype':
        colToChange = kwargs.get('colToChange')
        dtypeChoice = kwargs.get('dtypeChoice')  # one of: 'int','float','string','date'

        if colToChange not in dfLocal.columns:
            raise KeyError(f"Column {colToChange} not found")

        dtype_map = {
            'int': lambda x: pd.to_numeric(x, errors='coerce').astype('Int64'),
            'float': lambda x: pd.to_numeric(x, errors='coerce').astype(float),
            'string': lambda x: x.astype(str),
            'date': lambda x: pd.to_datetime(x, errors='coerce')
        }
        dfLocal[colToChange] = dtype_map[dtypeChoice](dfLocal[colToChange])
        return dfLocal
    elif action == "changeDate":
        colToEdit = kwargs.get('colToEdit')
        components = kwargs.get('components', [])  # list of strings like ['year','month','day','hour','minute','second']

        if colToEdit not in dfLocal.columns:
            raise KeyError(f"Column {colToEdit} not found")

        if not pd.api.types.is_datetime64_any_dtype(dfLocal[colToEdit]):
            raise TypeError(f"Column {colToEdit} is not datetime type")

        for comp in components:
            if comp == 'year':
                dfLocal[f"{colToEdit}_year"] = dfLocal[colToEdit].dt.year
            elif comp == 'month':
                dfLocal[f"{colToEdit}_month"] = dfLocal[colToEdit].dt.month
            elif comp == 'day':
                dfLocal[f"{colToEdit}_day"] = dfLocal[colToEdit].dt.day
            elif comp == 'hour':
                dfLocal[f"{colToEdit}_hour"] = dfLocal[colToEdit].dt.hour
            elif comp == 'minute':
                dfLocal[f"{colToEdit}_minute"] = dfLocal[colToEdit].dt.minute
            elif comp == 'second':
                dfLocal[f"{colToEdit}_second"] = dfLocal[colToEdit].dt.second
            else:
                raise ValueError(f"Unknown component {comp}")
        return dfLocal

    else:
        raise ValueError("Unknown action")

if __name__ == "__main__":
    main()