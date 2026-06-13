# DataReader Web App
Created by: Aiden Cabrera  
Purpose: To Streamline CSV data analysis for the Ramapo Climate Research Group

## Overview
The DataReader Web App is a Streamlit tool for cleaning, editing, analyzing, visualizing, and exporting tabular data. It remains a general CSV, Excel, and TXT reader, with a default NOAA daily climate preset for common research workflows used by the Ramapo Climate Research Group.

## Features
### Data Upload
- Supports `.csv`, `.xlsx`, `.xls`, and `.txt` files.
- Includes a NOAA daily climate preset and a generic data preset.
- Supports comma, tab, whitespace, semicolon, exact-space, and custom separators.
- Automatically detects headers when present and creates default column names when headers are missing.

### Cleaning & Preprocessing
- Converts numeric columns stored as strings to numeric types.
- Converts likely date columns automatically, with NOAA-safe defaults for climate datasets.
- Handles common missing-value markers and NOAA measurement placeholders.
- Preserves text fields for categorical analysis instead of aggressively normalizing casing.

### Data Exploration
- Numeric statistics: count, mean, standard deviation, min/max rows, and percentiles.
- Categorical statistics: unique counts and value frequencies.
- Data type tools for converting columns to date, float, integer, or string.

### Editing Data
- Rename or delete columns.
- Delete rows manually or by numeric condition.
- Drop duplicates by selected columns.
- Sort by column.
- Extract components of datetime columns.

### Visualization
- Embedded Continuous Wavelet Transform analysis with editable title and advanced settings.
- PNG download for generated wavelet graphs.

### Export
- Download the cleaned and edited working dataset as CSV.

Make edits and download the resulting CSV.

## Usage
Run the app locally:

```bash
streamlit run DataReaderWeb.py
```

Upload a dataset, choose the appropriate preset, edit or filter the working copy, then export the resulting CSV.

## Notes
All transformations are applied to a working copy; the original upload remains intact.
The download feature uses a browser-based save dialog, ensuring users can save locally without needing access to the server filesystem.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code as long as proper credit is given. See **LICENSE**.
