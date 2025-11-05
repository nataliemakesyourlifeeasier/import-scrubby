import pandas as pd
import numpy as np
import streamlit as st
import io
import json
import re
from pathlib import Path
from datetime import timedelta
import sys

# --- 1. CONFIGURATION (Embedded Validation Rules) ---

# The validation rules are embedded directly into the script for Streamlit deployment.
VALIDATION_RULES_CONFIG = {
  "validation_rules": [
    {"column": "firstName", "required": True, "type": "string", "description": "Guest's first name."},
    {"column": "lastName", "required": True, "type": "string", "description": "Guest's last name."},
    {"column": "Email", "required": True, "type": "string", "format": "email", "description": "Guest's email address."},
    {"column": "phone1", "required": True, "type": "string", "description": "Primary phone number."},
    {"column": "address1", "required": True, "type": "string", "description": "Primary address."},
    {"column": "postalCode", "required": True, "description": "postal code."},
    {"column": "city", "required": True, "type": "string", "description": "City."},
    {"column": "state", "required": True, "type": "string", "description": "state."},
    {"column": "country", "required": True, "type": "string", "description": "country."},
    {"column": "startDate", "required": True, "type": "date", "format": "%Y-%m-%d", "description": "Reservation start date (format: YYYY-MM-DD)."},
    {"column": "endDate", "required": True, "type": "date", "format": "%Y-%m-%d", "description": "Reservation end date (format: YYYY-MM-DD)."},
    {"column": "chargeTotal", "required": True, "type": "numeric", "min_value": 0, "description": "Total charge amount (Now Required)."},
    {"column": "paymentTotal", "required": True, "type": "numeric", "min_value": 0, "description": "Total payment amount (Now Required)."},
    {"column": "oldConfirmationNumber", "required": True, "description": "Previous confirmation number, if applicable (Now Required)."},
    {"column": "typeName", "required": True, "type": "string", "description": "The name of the reservation type or site type (must be provided)."},
    {"column": "siteName", "required": True, "type": "string", "description": "The specific site number or name (Now Required)."},
    {"column": "ageCategory0", "required": True, "type": "numeric", "min_value": 0, "description": "Count for age category 0 (e.g., Children) (Now Required)."},
    {"column": "ageCategory1", "required": True, "type": "numeric", "min_value": 0, "description": "Count for age category 1 (e.g., Adults) (Now Required)."},
    {"column": "ageCategory2", "required": True, "type": "numeric", "min_value": 0, "description": "Count for age category 2 (Now Required)."},
    {"column": "ageCategory3", "required": True, "type": "numeric", "min_value": 0, "description": "Count for age category 3 (Now Required)."},
    {"column": "pets", "required": True, "type": "numeric", "min_value": 0, "description": "The number of pets associated with the reservation (Now Required)."}
  ]
}

# --- 2. SCRUBBING LOGIC (from resPrettyer.py) ---

def scrub_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Performs core data cleanup: strips whitespace, drops duplicates/empty rows."""
    
    # CRITICAL FIX: The file is often imported with the header row treated as data.
    # We drop the first row to ensure validation starts on actual records.
    if len(df) > 1:
        df = df.iloc[1:].reset_index(drop=True)

    # Trim whitespace from all string cells
    # NOTE: df.applymap is used for compatibility with the old scrubber's behavior
    df = df.applymap(lambda x: str(x).strip() if pd.notnull(x) else x)

    # Drop rows that are entirely empty (only drops if cells are truly NaN/None, not just '')
    df = df.dropna(how="all").reset_index(drop=True)

    # Drop duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    return df

# --- 3. VALIDATION UTILITIES (from validator.py and date overlap logic) ---

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names to lowercase with no spaces for case-insensitive matching."""
    new_columns = {}
    for col in df.columns:
        col_str = str(col)
        # Convert to lowercase and remove spaces/special characters
        normalized_name = re.sub(r'[^a-zA-Z0-9]+', '', col_str).lower()
        new_columns[col] = normalized_name
        
    return df.rename(columns=new_columns)


def check_date_overlaps(df: pd.DataFrame) -> list:
    """Checks for overlapping reservations for the same campsiteId."""
    
    # NOTE: These dates must be normalized to datetime before this function is called
    df_valid = df.dropna(subset=['startdate', 'enddate'])
    df_valid = df_valid[df_valid['campsiteid'].notna()]
    
    overlap_errors = []

    def find_overlaps_in_group(group):
        group = group.sort_values('startdate').reset_index(drop=True)
        overlaps = []
        
        for i in range(len(group) - 1):
            res_a = group.iloc[i]
            res_b = group.iloc[i + 1]
            
            # Check for overlap: Reservation B starts before Reservation A ends
            if res_b['startdate'] < res_a['enddate']:
                overlaps.append({
                    'campsiteId': res_a['campsiteid'],
                    'start_A': res_a['startdate'].strftime('%Y-%m-%d'),
                    'end_A': res_a['enddate'].strftime('%Y-%m-%d'),
                    'start_B': res_b['startdate'].strftime('%Y-%m-%d'),
                    'end_B': res_b['enddate'].strftime('%Y-%m-%d'),
                })
        return overlaps

    if not df_valid.empty:
        # The apply operation requires the dataframe to have enough rows to group
        grouped_overlaps = df_valid.groupby('campsiteid').apply(find_overlaps_in_group)
        
        # Flatten the list of lists
        for overlaps_in_group in grouped_overlaps:
            overlap_errors.extend(overlaps_in_group)
            
    return overlap_errors


def validate_dataframe(df: pd.DataFrame, rules: list) -> list:
    """Applies validation rules and returns a list of error dictionaries."""
    errors = []
    
    # Ensure all empty strings are true NaN for universal checking
    df_check = df.replace(r'^\s*$', np.nan, regex=True).fillna(np.nan)

    for rule in rules:
        col_name = rule['column'].lower().replace(' ', '')
        report_col_name = rule['column'] # Original name for reporting
        
        # 1. Check for missing column
        if col_name not in df.columns:
            if rule.get('required'):
                errors.append({
                    "type": "MISSING_COLUMN",
                    "column": report_col_name,
                    "message": f"Required column '{report_col_name}' is missing from the file."
                })
            continue

        series = df_check[col_name]
        
        # 2. REQUIRED check
        if rule.get('required'):
            null_mask = series.isnull()
            if null_mask.any():
                error_indices = series[null_mask].index.tolist()
                for idx in error_indices:
                    errors.append({
                        "type": "MISSING_VALUE",
                        "column": report_col_name,
                        # Row calculation: +1 for 0-index, +1 for header row = +2
                        "row": idx + 2, 
                        "message": f"Required field is empty."
                    })

        # Filter out NaN values before running other checks
        series_clean = series.dropna()
        if series_clean.empty:
            continue
            
        # 3. TYPE and FORMAT checks
        
        # NUMERIC checks
        if rule.get('type') == 'numeric':
            try:
                numeric_series = pd.to_numeric(series_clean, errors='coerce')
                
                # Report non-numeric errors
                invalid_mask = numeric_series.isna()
                if invalid_mask.any():
                    error_indices = series_clean[invalid_mask].index.tolist()
                    for idx in error_indices:
                        errors.append({
                            "type": "TYPE_ERROR", "column": report_col_name, "row": idx + 2, 
                            "value": df.loc[idx, col_name], "message": f"Value is not a valid number."
                        })
                
 # Check min_value
                min_val = rule.get('min_value')
                if min_val is not None:
                    # FIX APPLIED HERE: Add a fail-safe check for the comparison
                    valid_numeric_series = numeric_series.dropna() 
                    
                    # Convert min_val to numeric (it's safe as it comes from JSON)
                    min_val = float(min_val) 
                    
                    # Ensure series contains only numeric types before comparison
                    valid_numeric_series = pd.to_numeric(valid_numeric_series, errors='coerce').dropna()
                    
                    if not valid_numeric_series.empty:
                        min_mask = valid_numeric_series < min_val
                        if min_mask.any():
                            error_indices = valid_numeric_series[min_mask].index.tolist()
                            for idx in error_indices:
                                errors.append({
                                    "type": "VALUE_ERROR", "column": report_col_name, "row": idx + 2,
                                    "value": df.loc[idx, col_name], "message": f"Value must be greater than or equal to {min_val}."
                                })

            except Exception:
                pass # Already handled by other checks

        # DATE checks
        elif rule.get('type') == 'date':
            date_format = rule.get('format')
            date_series = pd.to_datetime(series_clean, format=date_format, errors='coerce')
            invalid_mask = date_series.isna()
            if invalid_mask.any():
                error_indices = series_clean[invalid_mask].index.tolist()
                for idx in error_indices:
                    errors.append({
                        "type": "FORMAT_ERROR", "column": report_col_name, "row": idx + 2,
                        "value": df.loc[idx, col_name], "message": f"Date must match format '{date_format}'."
                    })
            # Add the converted date column back for the date overlap check
            df[col_name] = date_series 


        # EMAIL checks
        elif rule.get('format') == 'email':
            email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            invalid_mask = ~series_clean.str.match(email_regex, na=False)
            if invalid_mask.any():
                error_indices = series_clean[invalid_mask].index.tolist()
                for idx in error_indices:
                    errors.append({
                        "type": "FORMAT_ERROR", "column": report_col_name, "row": idx + 2,
                        "value": df.loc[idx, col_name], "message": f"Value is not a valid email address format."
                    })
    
    # 4. ADVANCED CHECK: Date Overlap
    if all(col in df.columns for col in ['startdate', 'enddate', 'campsiteid']):
        overlap_errors = check_date_overlaps(df)
        if overlap_errors:
            for error in overlap_errors:
                errors.append({
                    "type": "BUSINESS_RULE", "column": 'campsiteId', "row": "N/A", 
                    "message": f"Overlap: Res A ({error['start_A']} to {error['end_A']}) conflicts with Res B ({error['start_B']} to {error['end_B']}) for Campsite {error['campsiteId']}."
                })
                
    return errors


# --- 4. STREAMLIT INTERFACE AND EXECUTION ---

def process_and_validate_data(df_input) -> tuple:
    """
    Combines scrubbing and all validation checks.
    Returns: (validation_passed: bool, processed_df: pd.DataFrame, error_report: list)
    """
    # 1. SCRUBBING
    # Normalize column names immediately (required by validation logic)
    df_normalized = normalize_column_names(df_input.copy())
    
    # Scrub the data
    df_scrubbed = scrub_dataframe(df_normalized)
    
    # 2. VALIDATION
    rules = VALIDATION_RULES_CONFIG.get('validation_rules', [])
    errors = validate_dataframe(df_scrubbed, rules)
    
    return not errors, df_scrubbed, errors


def main_app():
    st.set_page_config(page_title="Reservation Data Processor", layout="centered")
    st.title("🗂️ Data Scrubber and Validator")
    st.markdown("Upload a file to automatically **scrub** it (clean rows/duplicates) and **validate** it against business rules.")

    uploaded_file = st.file_uploader("Choose an Excel or CSV file to process", type=['xlsx', 'xls', 'csv'])

    if uploaded_file is not None:
        try:
            # Read the file based on its type
            file_type = uploaded_file.name.split('.')[-1].lower()
            if file_type in ['xlsx', 'xls']:
                # Read without header=0 for the scrubber to correctly pick up headers
                df_input = pd.read_excel(uploaded_file, dtype=str)
            elif file_type == 'csv':
                df_input = pd.read_csv(uploaded_file, dtype=str)
            else:
                st.error("Unsupported file type.")
                return

            # --- Run the combined process ---
            validation_passed, processed_df, error_report = process_and_validate_data(df_input.copy())
            
            # --- Display Results ---
            st.subheader("1. Processing Report")
            st.info(f"Initial Rows: {len(df_input)} | Cleaned Rows: {len(processed_df)}")

            if validation_passed:
                st.success("🎉 ALL VALIDATION CHECKS PASSED. Data is ready for upload!")
                
                # Provide download button for the CLEANED data
                output = io.BytesIO()
                processed_df.to_excel(output, index=False)
                st.download_button(
                    label="⬇️ Download Cleaned File (Ready for Upload)",
                    data=output.getvalue(),
                    file_name=f"CLEANED_{uploaded_file.name}",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error(f"🛑 VALIDATION FAILED. Found {len(error_report)} error(s). Please correct the errors below.")
                
                # Display detailed error report
                st.subheader("2. Detailed Error Report")
                
                # Sort errors for better readability (by row, then column)
                error_report.sort(key=lambda e: (e.get('row', 'Z'), e['column']))
                
                for error in error_report:
                    row_info = f"Row {error.get('row', 'N/A')}"
                    val_info = f"Value: '{error.get('value', 'N/A')}'"
                    
                    st.markdown(
                        f"**❌ [{error.get('type')}]** {error['message']}<br>"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;`{row_info}, Column: {error['column']}` {val_info}", 
                        unsafe_allow_html=True
                    )
                
                # Display cleaned data preview (still useful for debugging)
                st.subheader("3. Cleaned Data Preview (for debugging)")
                st.dataframe(processed_df.head(10))

        except Exception as e:
            st.error(f"An unexpected error occurred during file processing: {e}")

if __name__ == "__main__":
    main_app() #
