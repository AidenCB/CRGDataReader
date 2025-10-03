import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sys

def main():
    if (verifyFile()):
        return

    filename = sys.argv[1]
    if (filename[-3:].lower() in ['csv', 'txt']):
        df = pd.read_csv(filename, header=None, sep=None, engine='python')
    elif (filename[-3:].lower() in ['xls', 'xlsx']):
        df = convertXLS(filename)

    # Store filename in dataframe attributes
    df.attrs['filename'] = filename
    
    # Set the header
    if checkHeader(df):
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
    else:
        df.columns = range(len(df.columns)) # To set header to just be integers

    # Remove duplicates based on all columns and reset index
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    # Cleans data after setting header
    df = cleanData(df)

    loop = True
    while loop:
        userChoice = input(
        "\n1. View/Edit Dataframe" # Opens menu to view/edit dataframe
        "\n2. Show Math Information" # Opens menu to show mathematical information depending if numeric or categorical
        "\n3. Visualize Data" # Uses matplotlib to create graphs on data
        "\n4. Save Dataframe" # Saves dataframe to new file or overwrites existing file 
        "\n5. Show Date Information" # Shows information for date columns if they exist
        "\n6. Exit"
        "\nInput: "
        )
        if userChoice.isdigit():
            userChoice = int(userChoice)
            if userChoice == 1:
                df = editData(df)
            elif userChoice == 2:
                showMathInfo(df)
            elif userChoice == 3:
                visualizeData(df)
            elif userChoice == 4:
                saveData(df)
            elif userChoice == 5:
                displayDates(df)
            elif userChoice == 6:
                loop = False
        else:
            print("Input only a number 1-6")

# Returns True if file is not 'good' to use
def verifyFile():
    if (len(sys.argv) >= 3):
        print("Too many arguments passed to command line.")
        return True
    elif (len(sys.argv) < 2):
        print("Not enough arguments passed to command line.")
        return True
    if (sys.argv[1][-3:].lower() not in ['csv', 'xls', 'xlsx', 'txt']):
        print("Enter only a csv, or xls/xlsx file.")
        return True        
    
# Converts a spreadsheet file to a csv file
def convertXLS(filename):
    file = pd.read_excel(filename)
    filename = filename[:-3] + "csv"
    file.to_csv(filename, header=None)
    df = pd.read_csv(filename, header=None, sep=None, engine='python')
    return df

# Checks first 2 rows of data frame to determine if there is a header
def checkHeader(mainDf):
    df = mainDf.copy()
    if len(df) < 1:
        return False
    
    totalVals = len(df.iloc[0])
    stringVals = sum(isinstance(val, str) for val in df.iloc[0])

    # Ratio check with threshold of 75%
    if (stringVals / totalVals) < .85:
        return False

    if len(df) < 2:
        return False

    secondRowStrings = sum(isinstance(val, str) for val in df.iloc[1])

    # If second row is under the threshold we can assume first row is header
    if secondRowStrings / totalVals < .5:
        return True
    
    # First row and second row are all strings
    # Need to compare to ensure that first row doesn't have same values as second row
    if df.iloc[0].equals(df.iloc[1]):
        # Row 1 = Row 2
        return False
    else:
        # Row 1 != Row 2
        return True

# Parses a column to see if it can be converted into datetime
def dateTimeColumn(series):   
    # Copy so we don’t mutate input directly
    s = series.copy()

    dateFormats = ["%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%d-%m-%Y",
                   "%m/%d/%y", "%m-%d-%y", "%d/%m/%y", "%d-%m-%y"]

    timeFormats = ["%H:%M:%S", "%H:%M", "%H-%M-%S", "%H-%M",
                   "%H.%M.%S", "%H.%M"]

    genericFormats = [
        "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m-%d-%Y %H:%M:%S", "%m-%d-%Y %H:%M",
        "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
        "%m/%d/%y %H:%M:%S", "%m/%d/%y %H:%M", "%m-%d-%y %H:%M:%S", "%m-%d-%y %H:%M",
        "%d/%m/%y %H:%M:%S", "%d/%m/%y %H:%M", "%d-%m-%y %H:%M:%S", "%d-%m-%y %H:%M", "%YYYY"
    ]

    successful = False

    # Try based on column name hint
    colname = s.name.lower() if isinstance(s.name, str) else ""

    if "date" in colname:
        for fmt in genericFormats + dateFormats:
            converted = pd.to_datetime(s, format=fmt, errors="coerce")
            if not converted.isna().all():
                s = converted
                successful = True
                break

    elif "time" in colname:
        for fmt in timeFormats:
            converted = pd.to_datetime(s, format=fmt, errors="coerce")
            if not converted.isna().all():
                s = converted.dt.time
                successful = True
                break

    # Try all generic formats if nothing worked yet
    if not successful:
        for fmt in genericFormats + dateFormats + timeFormats:
            converted = pd.to_datetime(s, format=fmt, errors="coerce")
            if not converted.isna().all():
                if "H" in fmt:   # crude check → format includes time
                    try:
                        s = converted.dt.time
                    except Exception:
                        s = converted
                else:
                    s = converted
                successful = True
                break

    # If still failed, return original
    if not successful:
        return series

    return s

# Function to test columns against different date formats
def getDateTime(copyDf):
    df = copyDf.copy()
    dateColumns = []

    for column in df.select_dtypes(include="object").columns:
        original = df[column].copy()
        converted = dateTimeColumn(df[column])

        # If conversion actually changed the data (not all NaT)
        if not converted.equals(original):
            df[column] = converted
            dateColumns.append(column)

    # Rename date columns
    if len(dateColumns) == 1:
        df = df.rename(columns={dateColumns[0]: 'date'})
    elif len(dateColumns) > 1:
        df = df.rename(columns={dateColumns[0]: 'date1'})

    df.attrs['date_columns'] = dateColumns

    return df

# Cleans the data of the dataframe
# Clears special characters (saves only: space, numbers, and characters)
# Clears empty rows with >50% missing data, fills data for less
def cleanData(mainDf):
    df = mainDf.copy()

    # Remove commas from numbers and replacing with periods
    for column in df.select_dtypes(include="object").columns:  # Only doing operation on strings
        df[column] = df[column].str.replace(',', '.', regex=True)

    # Check if strings contain "date" characters
    for column in df.select_dtypes(include="object").columns: 
    # Check for "/", "-" 
        if df[column].str.contains(r'[\/\-]').any():
            df[column] = dateTimeColumn(df[column])

    # Convert strings to numbers
    # Do it FIRST because it might convert numbers to date incorrectly otherwise
    for column in df.columns:
        if df[column].dtype == 'object':
            converted = pd.to_numeric(df[column], errors='coerce')
            # Does NOT have NaN, convert to numeric
            if not (converted.isna().all()):
                df[column] = converted 

    numericColumns = df.select_dtypes(include=['number']).columns

    # Convert strings to dates
    df = getDateTime(df)

    # Replacing common placeholder values
    placeholders = [-999, 999, -9999, 9999, 'NA', 'NaN', 'null', 'None', '', 'missing', -200]
    for col in df.columns:
        for pch in placeholders:
            df[col] = df[col].replace(to_replace=pch, value=np.nan)

   # Drop rows and columns with >=50% missing values
    df = df.dropna(axis=0, thresh=(len(df.columns) // 2))
    df = df.dropna(axis=1, thresh=(len(df.columns) // 2))

    # Fill missing values
    for column in df.columns:
        columnType = df[column].dtype.kind
        #if columnType in 'if':
            # df[column] = df[column].fillna(df[column].mean())
        if columnType == 'O':
            df[column] = df[column].fillna('unknown')
        elif columnType == 'M':
            df[column] = df[column].fillna(pd.Timestamp("1900-01-01"))

    # Cleaning strings (removing special characters, making lowercase)
    for column in df.select_dtypes(include="object").columns:  # Only doing operation on strings
            if pd.api.types.is_string_dtype(df[column]):
                df[column] = df[column].str.replace(r'[^\w\s]', '', regex=True)
                df[column] = df[column].str.lower()

    return df

# Functions to display various aspects of the dataframe
# Displays data for different dates
def displayDates(df):
    numericColumns = df.select_dtypes(include='number').columns
    if 'date' in df.columns:
        print(df.groupby(['date'])[numericColumns].mean().head())
    elif 'date1' in df.columns:
        print(df.groupby(['date1'])[numericColumns].mean().head())
        if 'date_columns' in df.attrs:
            for column in df.attrs['date_columns']:
                print(df.groupby([column])[numericColumns].mean().head())
    else:
        print("This dataset does not contain any date columns\n")

def displayUniques(df):
    for column in df.columns:
        print(f"Unique Values for {column}:")
        print(df[column].unique())

def visualizeData(df):
    numericColumns = df.select_dtypes(include=['number']).columns
    categoricalColumns = df.select_dtypes(exclude=['number']).columns

    # Histograms & Boxplots
    for column in numericColumns:
        plt.figure(figsize=(10, 4))

        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(df[column].dropna(), color='skyblue', edgecolor='black')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')

        # Boxplot
        plt.subplot(1, 2, 2)
        plt.boxplot(df[column].dropna(), vert=False)
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)

        plt.tight_layout()
        plt.show()

    # Bar charts
    for col in categoricalColumns:
        topValues = df[column].value_counts().head(10)
        plt.figure(figsize=(8, 4))
        plt.bar(topValues.index.astype(str), topValues.values, color='lightgreen', edgecolor='black')
        plt.title(f'Bar Chart of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Correlation Heatmap (Numeric Only)
    if len(numericColumns) > 1:
        plt.figure(figsize=(10, 8))
        correlation = df[numericColumns].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()

# Menu function to show either numerical or categorical statistics
def showMathInfo(df):
    def minMaxRow():
        minVal = dfStats.at['min', col]
        maxVal = dfStats.at['max', col]
        minRow = df[df[col] == minVal]
        maxRow = df[df[col] == maxVal]
        if not minRow.empty:
            print(f"\nMinimum Value, Located ({minVal}, {col}):")
            print(minRow)
        if not maxRow.empty:
            print(f"\nMaximum Value, Located ({maxVal}, {col}):")
            print(maxRow)

    loop = True
    while loop:
        viewNumStats = True
        userChoice = input(
        "\n1. Select Numeric Statistics"
        "\n2. Show Categorical Statistics"
        "\n3. Show Datatypes" 
        "\n4. Show Unique Values per Column"
        "\n5. Exit"
        "\nInput: ")
        if userChoice.isdigit():
            userChoice = int(userChoice)
            if userChoice == 1: # Show basic statistics 
                dfStats = df.describe(include=['number']) # Numeric columns then object columns
                while viewNumStats:
                    print("\nNumeric Statistics Options:")
                    userChoice = input(
                    "\n1. Show number per column"
                    "\n2. Show mean"
                    "\n3. Show std"
                    "\n4. Show min/max"
                    "\n5. Show percentile"
                    "\n6. Show all statistics"
                    "\n7. Exit"
                    "\nInput: ")
                    if (userChoice.isdigit()):
                        userChoice = int(userChoice)
                        if userChoice == 1:
                            print("\nCount Values:")
                            print(dfStats.loc['count'])
                        elif userChoice == 2:
                            print("\nMean Values:")
                            print(dfStats.loc['mean'])
                        elif userChoice == 3:
                            print("\nStandard Deviation Values:")
                            print(dfStats.loc['std'])
                        elif userChoice == 4:
                            # print("\nMinimum Values:")
                            # print(dfStats.loc['min'])
                            # print("\nMaximum Values:")
                            # print(dfStats.loc['max'])
                            # Finds the object column associated with the min/max value
                            # Also works with datetime columns

                            # User selects which column to find min/max for by using its header name or number
                            print(f"Current Columns: {list(df.select_dtypes(include=['number']).columns)}")
                            userCol = input("Which column to find min/max rows for? (type 'all' to show for all numeric columns)\nInput:"
                            "\nInput:")
                            if (userCol.isdigit()):
                                userCol = int(userCol)
                                for col in df.select_dtypes(include=['number']).columns:
                                    if col == userCol or df.columns.get_loc(col) == userCol:
                                        minMaxRow()
                            elif isinstance(userCol, str):
                                userCol = userCol.strip().lower()
                                for col in df.select_dtypes(include=['number']).columns:
                                    if col == userCol:
                                        minMaxRow()

                            elif userCol == 'all':
                                if ((df.select_dtypes(include=['datetime']).shape[1] > 0) or (df.select_dtypes(include=['object']).shape[1] > 0)):
                                    for col in df.select_dtypes(include=['number']).columns:
                                        minMaxRow()

                        elif userChoice == 5:
                            userPercent = input("What percentile to show? (0-100): ")
                            if (userPercent.replace('.','',1).isdigit()): # Checks if a number is entered and allows for decimals
                                userPercent = float(userPercent)
                                userPercent = round(userPercent, 2)
                                if 0 <= userPercent <= 100:
                                    userPercent = userPercent / 100
                                    dfPercent = df.quantile(userPercent, numeric_only=True)
                                    print(f"\n{userPercent} Percentile Values:")
                                    print(dfPercent)
                                else:
                                    print("Input only a number between 0-100")
                        elif userChoice == 6:
                            print("\nAll Statistics:")
                            print(dfStats)
                        elif userChoice == 7:
                            viewNumStats = False
                        else:
                            print("Input only a number 1-10")
            elif userChoice == 2: # Show categorical statistics
                dfStats = df.describe(include=['object']) 
                viewStats = True
                while viewStats:
                    print("\nCategorical Statistics Options:")
                    userChoice = input(
                    "\n1. Show number per column"
                    "\n2. Show unique values"
                    "\n3. Show top value"
                    "\n4. Show frequency of top value"
                    "\n5. Show all statistics"
                    "\n6. Exit"
                    "\nInput: ")
                    if (userChoice.isdigit()):
                        userChoice = int(userChoice)
                        if userChoice == 1:
                            print("\nCount Values:")
                            print(dfStats.loc['count'])
                        elif userChoice == 2:
                            print("\nUnique Values:")
                            print(dfStats.loc['unique'])
                        elif userChoice == 3:
                            print("\nTop Values:")
                            print(dfStats.loc['top'])
                        elif userChoice == 4:
                            print("\nFrequency of Top Values:")
                            print(dfStats.loc['freq'])
                        elif userChoice == 5:
                            print("\nAll Statistics:")
                            print(dfStats)
                        elif userChoice == 6:
                            viewStats = False
                        else:
                            print("Input only a number 1-6")
            elif userChoice == 3:
                print(df.dtypes)
            elif userChoice == 4:
                print(df.nunique())
            elif userChoice == 5:
                loop = False
            else:
                print("Input only a number 1-5")
        else:
            print("Input only a number 1-5")

# Menu function to edit and view the dataframe
def editData(df):
    loop = True
    while loop:
        userChoice = input(
        "\n1. View head of dataframe"
        "\n2. Edit Column Names"
        "\n3. Delete Column"
        "\n4. Edit Datatypes"
        "\n5. Exit"
        "\nInput: ")
        if userChoice.isdigit():
            userChoice = int(userChoice)
            if userChoice == 1:
                numShow = input("How many rows to show from the top? (default 5): ")
                if not numShow.isdigit():
                    numShow = 5
                else:
                    numShow = int(numShow)
                print(df.head(numShow))
            elif userChoice == 2:
                print(f"Current Columns: {list(df.columns)}")
                colToEdit = input("Enter the exact column name to edit: ")
                if (colToEdit.isdigit()):
                    colToEdit = int(colToEdit)
                if (colToEdit) in df.columns:
                    newColName = input("Enter the new column name: ")
                    df = df.rename(columns={colToEdit: newColName})
                    print(f"Column '{colToEdit}' renamed to '{newColName}'")
                else:
                    print(f"Column '{colToEdit}' not found in dataframe.")
            elif userChoice == 3:
                colToDelete = input("Enter the exact column name to delete: ")
                if colToDelete in df.columns:
                    df = df.drop(columns=[colToDelete])
                    print(f"Column '{colToDelete}' deleted from dataframe.")
                else:
                    print(f"Column '{colToDelete}' not found in dataframe.")
            elif userChoice == 4:
                print(f"Current Columns and Datatypes:\n{df.dtypes}")
                colToChange = input("Enter the exact column name to change datatype: ")
                if (colToChange.isdigit()):
                    colToChange = int(colToChange)
                if colToChange in df.columns:
                    dtypeChoice = input("Select new datatype:"
                          "\n1. Integer"
                          "\n2. Float"
                          "\n3. String"
                          "\n4. Date"
                          "\nInput: ")
                    if dtypeChoice.isdigit():
                        dtypeChoice = int(dtypeChoice)
                        try:
                            if dtypeChoice == 1:
                                df[colToChange] = pd.to_numeric(df[colToChange], errors='coerce').astype('Int64')
                            elif dtypeChoice == 2:
                                df[colToChange] = pd.to_numeric(df[colToChange], errors='coerce').astype(float)
                            elif dtypeChoice == 3:
                                df[colToChange] = df[colToChange].astype(str)
                            elif dtypeChoice == 4:
                                timeChoice = input(
                                    "\n1. Year"
                                    "\n2. Month"
                                    "\n3. Day"
                                    "\n4. Hour:Minute:Second"
                                )
                                if timeChoice.isdigit():
                                    timeChoice = int(timeChoice)
                                    if timeChoice == 1:
                                        df[colToChange] = pd.to_datetime(df[colToChange], format="%Y", errors='coerce')
                                    elif timeChoice == 2:
                                        df[colToChange] = pd.to_datetime(df[colToChange], format="%m/%Y", errors='coerce')
                                    elif timeChoice == 3:
                                        df[colToChange] = pd.to_datetime(df[colToChange], format="%m/%d/%Y", errors='coerce')
                                    elif timeChoice == 4:
                                        df[colToChange] = pd.to_datetime(df[colToChange], errors='coerce')
                                    else:
                                        print("Input only a number 1-4")
                            else:
                                print("Input only a number 1-4")
                        except Exception as e:
                            print(f"Error converting column: {e}")
                    else:
                        print("Input only a number 1-4")
                else:
                    print(f"Column '{colToChange}' not found in dataframe.")
            elif userChoice == 5:
                loop = False
        else:
            print("Input only a number 1-3")
    return df

def saveData(df):
    userChoice = input(f"Do you want to save the dataframe to a new file (y), or overwrite the existing file (o)? (y/o/n): ")
    if userChoice.lower() == 'y':
        while True:
            newFilename = input("Enter new filename (with .csv extension): ")
            if not newFilename.endswith('.csv'):
                print("Filename must end with .csv")
            else:
                break
        df.to_csv(newFilename, index=False)
        print(f"Dataframe saved to {newFilename}")
    elif userChoice.lower() == 'o':
        if 'filename' in df.attrs:
            originalFilename = df.attrs['filename']
            df.to_csv(originalFilename, index=False)
            print(f"Dataframe overwritten to {originalFilename}")
        else:
            print("Original filename not found. Cannot overwrite.") 
    else:
        print("Dataframe not saved.")

if __name__ == "__main__":
    main()
