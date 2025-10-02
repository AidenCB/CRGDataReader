import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# -----------------------
# Helper functions (ported from DataReader.py)
# -----------------------

def checkHeader(mainDf):
    df = mainDf.copy()
    if len(df) < 1:
        return False

    totalVals = len(df.iloc[0])
    stringVals = sum(isinstance(val, str) for val in df.iloc[0])

    if (stringVals / totalVals) < 0.85:
        return False
    else:
        return True

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

    # Fallback: try to let pandas infer format
    if not successful:
        try:
            converted = pd.to_datetime(s, errors='coerce', infer_datetime_format=True)
            if not converted.isna().all():
                s = converted
                successful = True
        except Exception:
            successful = False

    if not successful:
        return series
    return s

def getDateTime(copyDf):
    df = copyDf.copy()
    dateColumns = []
    
    # Only show multiselect if there are object columns
    objectCols = df.select_dtypes(include=['object']).columns.tolist()
    if objectCols and 'dateCols' not in st.session_state:
        # Store user selection in session state to avoid re-asking
        dateCols = st.multiselect(
            "Select which columns should be read as dates:",
            options=objectCols,
            key='date_column_selector'
        )
        if dateCols:
            st.session_state.dateCols = dateCols
    
    # Use stored selection if available
    if 'dateCols' in st.session_state:
        for col in st.session_state.dateCols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                dateColumns.append(col)
    
    # Rename date columns
    if len(dateColumns) == 1:
        df = df.rename(columns={dateColumns[0]: 'date'})
    elif len(dateColumns) > 1:
        df = df.rename(columns={dateColumns[0]: 'date1'})
    
    df.attrs['date_columns'] = dateColumns
    return df

def cleanData(mainDf):
    df = mainDf.copy()

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
        if not (converted.isna().all):
            df[column] = converted 


    # Convert strings to dates
    df = getDateTime(df)

    # Replace common placeholders with NaN
    placeholders = [-999, 999, -9, 9999, 'NA', 'NaN', 'null', 'None', '', 'missing', -200]
    for col in df.columns:
        for pch in placeholders:
            df[col] = df[col].replace(to_replace=pch, value=np.nan)

    # Drop rows/columns with >= 75% missing values 
    df = df.dropna(axis=0, thresh=(len(df.columns) * .2))
    df = df.dropna(axis=1, thresh=(len(df.columns) * .2))

    # Clean string columns: remove special characters, lowercase
    for column in df.select_dtypes(include="object").columns:
        if pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].str.replace(r'[^\w\s]', '', regex=True)
            df[column] = df[column].str.lower()

    return df

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



def displayUniques(df):
    uniques = {}
    for column in df.columns:
        uniques[column] = df[column].unique()
    return uniques

def visualizeData(df):
    # Note: In Streamlit we will call visual functions directly in UI.
    # This helper returns list of numeric columns and categorical columns
    numericColumns = df.select_dtypes(include=['number']).columns.tolist()
    categoricalColumns = df.select_dtypes(exclude=['number']).columns.tolist()
    return numericColumns, categoricalColumns

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

# -----------------------
# Streamlit UI and mapping all functions
# -----------------------

st.set_page_config(page_title="DataReader Web", layout="wide")
st.title("Datareader Web App")
st.write("Created by Aiden Cabrera for the Ramapo Climate Research Group")

# File uploader
uploadedFile = st.file_uploader("Upload CSV, Excel, or TXT file", type=["csv", "xlsx", "xls", "txt"])
dfRaw = None

# Reset / Clear session button
if st.button("Reset Data"):
    for key in ["lastFilename", "dfRaw", "workingDf"]:
        if key in st.session_state:
            del st.session_state[key]
    st.success("Session state cleared. Upload a new file to start again.")

if uploadedFile is not None:
    # Always reload on new file
    if 'lastFilename' not in st.session_state or st.session_state.lastFilename != uploadedFile.name:
        try:
            if uploadedFile.name.endswith((".csv", ".txt")):
                dfRaw = pd.read_csv(uploadedFile, header=None) 
            elif uploadedFile.name.endswith((".xlsx", ".xls")):
                dfRaw = pd.read_excel(uploadedFile)
            else:
                raise ValueError("Unsupported file type")

            st.success("File uploaded successfully.")

            # Store original filename
            st.session_state.lastFilename = uploadedFile.name
            dfRaw.attrs['filename'] = uploadedFile.name
            st.session_state.dfRaw = dfRaw.copy()

                # If header exists, convert first row into header
            if checkHeader(dfRaw):
                # Convert first row values to lowercase if they are strings
                fixedCol = []
                for val in dfRaw.iloc[0]:
                    if isinstance(val, str):
                        fixedCol.append(val.strip().lower())
                    else:
                        fixedCol.append(val)

                dfRaw.columns = fixedCol
                dfRaw = dfRaw.drop(index=0)
                dfRaw = dfRaw.reset_index(drop=True)  # Remove the old header row, reset index
            else:
                # create default column names
                dfRaw.columns = [f"col_{i}" for i in range(len(dfRaw.columns))]

            # Clean only once
            st.session_state.workingDf = cleanData(dfRaw)

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.session_state.workingDf = None

    # Always work with session copy
    workingDf = st.session_state.get("workingDf")


    st.sidebar.subheader("Main Menu")
    mainMenu = st.sidebar.selectbox(
        "Select operation",
        [
            "Numeric Statistics",
            "Categorical Statistics",
            "Datatypes",
            "Display Dates",
            "Display Uniques",
            "Clean Data",
            "Edit Dataframe",
            "Visualize Data",
            "Save / Export"
        ]
    )

    # Preview
    with st.expander("Preview data (first 10 rows)"):
        st.dataframe(workingDf.head(10))

    # ---------- Numeric Statistics ----------
    if mainMenu == "Numeric Statistics":
        numericCols = workingDf.select_dtypes(include=['number']).columns.tolist()
        if not numericCols:
            st.info("No numeric columns available")
        else:
            st.subheader("Numeric Statistics")
            dfStats = showMathInfo(workingDf)

            statOption = st.radio(
                "Choose a numeric statistic to view",
                ["Count per column", "Mean", "Std", "Min/Max rows", "Percentile", "All statistics"]
            )

            if statOption == "Count per column":
                st.write(dfStats.loc['count'])

            elif statOption == "Mean":
                st.write(dfStats.loc['mean'])

            elif statOption == "Std":
                st.write(dfStats.loc['std'])

            elif statOption == "Min/Max rows":
                else:
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
                        else:
                            st.write("No min row found")
                        if not maxRow.empty:
                            st.write("Max row(s):")
                            st.dataframe(maxRow)
                        else:
                            st.write("No max row found")
                    if colChoice == "All":
                        for col in numericCols:
                            minMaxRow(col)
                    else:
                        minMaxRow(colChoice)

            elif statOption == "Percentile":
                userPercent = st.number_input("Percentile (0-100)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
                if st.button("Show Percentile"):
                    percent = round(userPercent / 100.0, 4)
                    dfPercent = workingDf.quantile(percent, numeric_only=True)
                    st.write(f"{userPercent}% percentile values")
                    st.dataframe(dfPercent)

            elif statOption == "All statistics":
                st.dataframe(dfStats)

    # ---------- Categorical Statistics ----------
    elif mainMenu == "Categorical Statistics":
        st.subheader("Categorical Statistics")
        catStats = showCategoricalInfo(workingDf)
        st.dataframe(catStats)


    # ---------- Datatypes ----------
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

    # ---------- Display Dates ----------
    elif mainMenu == "Display Dates":
        st.subheader("Display date-grouped numeric means")
        dateOutputs = displayDates(workingDf)
        if not dateOutputs:
            st.info("No date columns found")
        else:
            for k, v in dateOutputs.items():
                st.write(f"Group by {k}")
                st.dataframe(v)

    # ---------- Display Uniques ----------
    elif mainMenu == "Display Uniques":
        st.subheader("Unique values per column")
        uniques = displayUniques(workingDf)
        for col, vals in uniques.items():
            st.write(f"{col} ({len(vals)} unique)")
            st.write(vals)

    # ---------- Clean Data ----------
    elif mainMenu == "Clean Data":
        st.subheader("Clean data options")
        st.write("Use these actions to clean the working dataframe. Changes are applied to the working copy.")

        if st.button("Reset working dataframe to original upload"):
            workingDf = st.session_state.workingDf
            st.success("Reset complete")
            st.dataframe(workingDf.head())

    # ---------- Edit Dataframe ----------
    elif mainMenu == "Edit Dataframe":
        st.subheader("Edit dataframe")
        editAction = st.selectbox("Choose edit action", ["View head", "Rename column", "Delete column", "Drop duplicates", "Sort by column", "Change Date"])
        if editAction == "View head":
            numShow = st.number_input("Rows to show", min_value=1, max_value=1000, value=5)
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
            colToDelete = st.selectbox("Select column to delete", workingDf.columns.tolist())
            if st.button("Delete column"):
                try:
                    workingDf = editData(workingDf, 'deleteColumn', colToDelete=colToDelete)
                    st.session_state.workingDf = workingDf
                    st.success(f"Deleted column {colToDelete}")
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

            # ---------- Visualize Data ----------
    elif mainMenu == "Visualize Data":
        st.subheader("Visualize data")
        numericCols, categoricalCols = visualizeData(workingDf)
        vizType = st.selectbox("Choose visualization", ["Histogram", "Boxplot", "Scatter", "Bar (categorical)", "Correlation heatmap"])
        if vizType == "Histogram":
            if not numericCols:
                st.info("No numeric columns available")
            else:
                colChoice = st.selectbox("Select numeric column", numericCols)
                bins = st.slider("Bins", min_value=5, max_value=200, value=20)
                fig, ax = plt.subplots()
                ax.hist(workingDf[colChoice].dropna(), bins=bins)
                ax.set_title(f"Histogram of {colChoice}")
                st.pyplot(fig)

        elif vizType == "Boxplot":
            if not numericCols:
                st.info("No numeric columns available")
            else:
                colChoice = st.selectbox("Select numeric column", numericCols)
                fig, ax = plt.subplots()
                ax.boxplot(workingDf[colChoice].dropna())
                ax.set_title(f"Boxplot of {colChoice}")
                st.pyplot(fig)

        elif vizType == "Scatter":
            if len(numericCols) < 2:
                st.info("Need at least two numeric columns for scatter plot")
            else:
                xCol = st.selectbox("X axis", numericCols)
                yCol = st.selectbox("Y axis", [c for c in numericCols if c != xCol])
                fig, ax = plt.subplots()
                ax.scatter(workingDf[xCol], workingDf[yCol])
                ax.set_xlabel(xCol)
                ax.set_ylabel(yCol)
                ax.set_title(f"Scatter: {xCol} vs {yCol}")
                st.pyplot(fig)

        elif vizType == "Bar (categorical)":
            if not categoricalCols:
                st.info("No categorical columns")
            else:
                colChoice = st.selectbox("Select categorical column", categoricalCols)
                counts = workingDf[colChoice].value_counts().head(50)
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=counts.values, y=counts.index, ax=ax)
                ax.set_xlabel("Count")
                ax.set_ylabel(colChoice)
                st.pyplot(fig)

        elif vizType == "Correlation heatmap":
            if len(numericCols) < 2:
                st.info("Need at least two numeric columns")
            else:
                corr = workingDf[numericCols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

    # ---------- Save / Export ----------
    elif mainMenu == "Save / Export":
        st.subheader("Save or export working dataframe")
        saveChoice = st.radio("Save options", ["New File", "Same file"])
        if saveChoice == "New File":
            newFilename = st.text_input("Enter new filename (end with .csv)")
            if not newFilename.endswith('.csv'):
                st.error("Filename must end with .csv")
            else:
                workingDf.to_csv(newFilename, index=False)
                st.success(f"Saved to {newFilename}")
        elif saveChoice == "Overwrite original file (if allowed)":
            if 'filename' in df.attrs:
                if st.button("Overwrite original upload"):
                    originalFilename = df.attrs.get('filename')
                    try:
                        workingDf.to_csv(originalFilename, index=False)
                        st.success(f"Overwrote {originalFilename}")
                    except Exception as e:
                        st.error(f"Could not overwrite: {e}")

    # Update session state workingDf
    st.session_state.workingDf = workingDf
