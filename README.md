#  DataReader Web App
Created by: Aiden Cabrera  
Purpose: To Streamline CSV data analysis for the Ramapo Climate Research Group

##  Overview
The DataReader Web App is a browser based tool, built with Python and hosted through Streamlit, for cleaning, analyzing, and visualizing data. 
It was developed to help members of the Ramapo Climate Research Group quickly process and explore the datasets they receive.
Users can upload CSV, Excel, or TXT files, inspect and clean the data, perform statistical analysis, visualize trends, and export the results.

##  Features
###  Data Upload:
-  Supports .csv, .xlsx, .xls, and .txt files.
-  Automatically detects headers when present.
-  Creates default column names if headers are missing.

###  Cleaning & Preprocessing:
-  Cleans string data (lowercases, removes special characters).
-  Converts numeric columns stored as strings to proper numeric types.
-  Converts date columns automatically and allows user specification.
-  Handles common placeholders for missing values (e.g., NA, -999, null).
-  Drops rows/columns with excessive missing values.

###  Data Exploration:
-  Numeric statistics: count, mean, standard deviation, min/max, percentiles.
-  Categorical statistics: unique values, frequency counts.
-  Data types: view and change column data types.

###  Editing Data:
-  Rename or delete columns:
-  Drop duplicates (customizable by column subset).
-  Sort by columns.
-  Extract components of datetime columns (year, month, day, hour, minute, second).

###  Visualization:
-  Histogram and boxplot for numeric columns.
-  Scatter plots for numeric comparisons.
-  Bar charts for categorical data.
-  Correlation heatmaps.

###  Export:
-  Download the cleaned and edited dataset as CSV:

###  Session Management:
-  Maintains a working copy in the browser session.
-  Optional reset button to clear all session data and start fresh.

##  Usage
Upload your CSV, Excel, or TXT file.
Preview your data in the main panel.
Use the sidebar to select operations:
-  Numeric Statistics
-  Categorical Statistics
-  Datatypes
-  Display Dates or Uniques
-  Clean Data
-  Edit Dataframe
-  Visualize Data
-  Save / Export

Make edits and download the resulting CSV.

##  Notes
All transformations are applied to a working copy; the original upload remains intact.
Date columns can be auto-detected or selected manually.
The download feature uses a browser-based save dialog, ensuring users can save locally without needing access to the server filesystem.

Licens

This project is licensed under the MIT License. You are free to use, modify, and distribute the code as long as proper credit is given. See **LICENSE**.
