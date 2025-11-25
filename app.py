import pandas as pd
import numpy as np
import json
import re
import sys
from pathlib import Path
from datetime import timedelta

# --- 1. CONFIGURATION (Embedded Validation Rules) ---

# The validation rules are embedded directly for execution.
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
    # Age Categories: 'required' removed here, handled by custom logic
    {"column": "ageCategory0", "type": "numeric", "min_value": 0, "description": "Count for age category 0 (e.g., Children)."},
    {"column": "ageCategory1", "type": "numeric", "min_value": 0, "description": "Count for age category 1 (e.g., Adults)."},
    {"column": "ageCategory2", "type": "numeric", "min_value": 0, "description": "Count for age category 2."},
    {"column": "ageCategory3", "type": "numeric", "min_value": 0, "description": "Count for age category 3."},
    # Pets: 'required' removed here, handled by custom logic
    {"column": "pets", "type": "numeric", "min_value": 0, "description": "The number of pets associated with the reservation."}
  ]
}

# --- 2. CORE LOGIC FUNCTIONS ---

def scrub_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Performs core data cleanup and applies defaulting rules."""
    
    # CRITICAL FIX: We drop the first row to ensure validation starts on actual records.
    if len(df) > 1:
        df = df.iloc[1:].reset_index(drop=True)

    # Trim whitespace from all string cells
    # NOTE: df.applymap is used for compatibility with the old scrubber's behavior
    df = df.applymap(lambda x: str(x).strip() if pd.notnull(x) else x)
    
    # 1. Convert empty cells to NaN for unified defaulting
    for col in df.columns:
        df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)

    # ---------------------------------------------------------------------
    # NEW RULE: GUARANTEED HEADCOUNT/PETS DEFAULTING
    # This applies '0' to ANY missing value in the headcount columns IF the column exists.
    # ---------------------------------------------------------------------
    headcount_cols = ['agecategory0', 'agecategory1', 'agecategory2', 'agecategory3', 'pets']
    
    for col in headcount_cols:
        if col in df.columns:
            # Fill ANY NaN in the column with '0'. This is the GUARANTEED fix.
            df[col] = df[col].fillna('0') 
            
    # Drop rows that are entirely empty (only drops if cells are truly NaN/None, not just filled with '0')
    df = df.dropna(how="all").reset_index(drop=True)

    # Drop duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    return df

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names to lowercase with no spaces for case-insensitive matching."""
    new_columns = {}
    for col in df.columns:
        col_str = str(col)
        normalized_name = re.sub(r'[^a-zA-Z0-9]+', '', col_str).lower()
        new_columns[col] = normalized_name
    return df.rename(columns=new_columns)

def check_date_overlaps(df: pd.DataFrame) -> list:
    """Checks for overlapping reservations for the same campsiteId."""
    
    df_valid = df.dropna(subset=['startdate', 'enddate'])
    df_valid = df_valid[df_valid['campsiteid'].notna()]
    
    overlap_errors = []

    def find_overlaps_in_group(group):
        group = group.sort_values('startdate').reset_index(drop=True)
        overlaps = []
        
        for i in range(len(group) - 1):
            res_a = group.iloc[i]
            res_b = group.iloc[i + 1]
            
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
        grouped_overlaps = df_valid.groupby('campsiteid').apply(find_overlaps_in_group)
        
        for overlaps_in_group in grouped_overlaps:
            overlap_errors.extend(overlaps_in_group)
            
    return overlap_errors


def validate_dataframe(df: pd.DataFrame, rules: list) -> list:
    """Applies validation rules and returns a list of error dictionaries."""
    errors = []
    df_check = df.replace(r'^\s*$', np.nan, regex=True).fillna(np.nan) 
    
    age_cols = ['ageCategory0', 'ageCategory1', 'ageCategory2', 'ageCategory3']

    # ---------------------------------------------------------------------
    # Custom Check for At Least One Age Category
    # ---------------------------------------------------------------------
    if all(col in df.columns for col in age_cols):
        # Check rows where ALL age columns are still NaN (meaning they were entirely missing before fillna('0'))
        all_age_missing_mask = df_check[age_cols].isna().all(axis=1) 

        if all_age_missing_mask.any():
            error_indices = df_check[all_age_missing_mask].index.tolist()
            for idx in error_indices:
                errors.append({
                    "type": "MISSING_HEADCOUNT",
                    "column": "Headcount Group",
                    "row": idx + 3,
                    "message": "All age categories are missing. At least one headcount value must be supplied."
                })


    for rule in rules:
        col_name = rule['column'].lower().replace(' ', '')
        report_col_name = rule['column'] 
        
        # 1. Check for missing column
        if col_name not in df.columns:
            if rule.get('required'):
                errors.append({"type": "MISSING_COLUMN", "column": report_col_name, "message": f"Required column '{report_col_name}' is missing from the file."})
            continue

        series = df_check[col_name]
        
        # 2. REQUIRED check (Only runs for columns where required=True in the JSON)
        if rule.get('required'):
            null_mask = series.isnull()
            if null_mask.any():
                error_indices = series[null_mask].index.tolist()
                for idx in error_indices:
                    errors.append({"type": "MISSING_VALUE", "column": report_col_name, "row": idx + 3, "message": f"Required field is empty."})

        series_clean = series.dropna()
        if series_clean.empty: continue
            
        # 3. TYPE and FORMAT checks
        
        # NUMERIC checks
        if rule.get('type') == 'numeric':
            try:
                numeric_series = pd.to_numeric(series_clean, errors='coerce')
                
                invalid_mask = numeric_series.isna()
                if invalid_mask.any():
                    error_indices = series_clean[invalid_mask].index.tolist()
                    for idx in error_indices:
                        errors.append({"type": "TYPE_ERROR", "column": report_col_name, "row": idx + 3, "value": df.loc[idx, col_name], "message": f"Value is not a valid number."})
                
                # Check min_value
                min_val = rule.get('min_value')
                if min_val is not None:
                    valid_numeric_series = numeric_series.dropna().astype(float) 
                    min_mask = valid_numeric_series < min_val
                    if min_mask.any():
                        error_indices = valid_numeric_series[min_mask].index.tolist()
                        for idx in error_indices:
                            errors.append({"type": "VALUE_ERROR", "column": report_col_name, "row": idx + 3, "value": df.loc[idx, col_name], "message": f"Value must be greater than or equal to {min_val}."})

            except Exception:
                pass 

        # DATE checks
        elif rule.get('type') == 'date':
            date_format = rule.get('format')
            date_series = pd.to_datetime(series_clean, errors='coerce') 
            invalid_mask = date_series.isna()
            if invalid_mask.any():
                error_indices = series_clean[invalid_mask].index.tolist()
                for idx in error_indices:
                    errors.append({"type": "FORMAT_ERROR", "column": report_col_name, "row": idx + 3, "value": df.loc[idx, col_name], "message": f"Date must be a recognizable date format."})
            df[col_name] = date_series 

        # EMAIL checks
        elif rule.get('format') == 'email':
            email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            invalid_mask = ~series_clean.str.match(email_regex, na=False)
            if invalid_mask.any():
                error_indices = series_clean[invalid_mask].index.tolist()
                for idx in error_indices:
                    errors.append({"type": "FORMAT_ERROR", "column": report_col_name, "row": idx + 3, "value": df.loc[idx, col_name], "message": f"Value is not a valid email address format."})
    
    # ADVANCED CHECK: Date Overlap
    if all(col in df.columns for col in ['startdate', 'enddate', 'campsiteid']):
        df['startdate'] = pd.to_datetime(df['startdate'], errors='coerce')
        df['enddate'] = pd.to_datetime(df['enddate'], errors='coerce')
        overlap_errors = check_date_overlaps(df)
        if overlap_errors:
            for error in overlap_errors:
                errors.append({"type": "BUSINESS_RULE", "column": 'campsiteId', "row": "N/A", "message": f"Overlap: Res A ({error['start_A']} to {error['end_A']}) conflicts with Res B ({error['start_B']} to {error['end_B']}) for Campsite {error['campsiteId']}."})
                
    return errors

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scrubber_validator_cli.py <input_excel_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    input_path = Path(input_file)
    rules = VALIDATION_RULES_CONFIG.get('validation_rules', [])

    print(f"--- Starting Data Processor for {input_file} ---")
    
    try:
        # Load the file
        if input_path.suffix.lower() in [".xls", ".xlsx"]:
            df_raw = pd.read_excel(input_path, dtype=str)
        else:
            df_raw = pd.read_csv(input_path, dtype=str)

        df_normalized = normalize_column_names(df_raw.copy())
        df_scrubbed = scrub_dataframe(df_normalized)
        
        # Execute validation
        errors = validate_dataframe(df_scrubbed, rules)

        print(f"Initial Rows: {len(df_raw)} | Cleaned Rows: {len(df_scrubbed)}")

        if not errors:
            # Generate output filename (e.g., input.xlsx -> CLEANED_input.csv)
            output_name = f"CLEANED_{input_path.stem}.csv"
            output_path = input_path.parent / output_name
            
            # Save to CSV
            df_scrubbed.to_csv(output_path, index=False)
            
            print(f"\n✅ Success! The file passed all validation rules.")
            print(f"The cleaned file has been saved to: {output_path}")
        else:
            print(f"\n❌ Validation FAILED. Found {len(errors)} error(s).")
            
            # --- DETAILED ERROR REPORT ---
            print("\n--- Detailed Error Report ---")
            
            # Crash-Proof Sorting Logic
            def stable_row_sort_key(e):
                row_value = e.get('row')
                try:
                    return int(row_value) 
                except (ValueError, TypeError):
                    return 999999 
            
            # Sort errors using the stable key
            errors.sort(key=lambda e: (stable_row_sort_key(e), e['column']))

            for error in errors:
                report_col_name = error['column']
                error_value = str(error.get('value', 'N/A'))
                val_info = f" (Value: '{error_value}')" if error_value != 'N/A' else ""
                
                print(f"[ROW {error.get('row', 'N/A')}, COL '{report_col_name}'] {error['message']}{val_info} (Type: {error['type']})")

    except Exception as e:
        print(f"\n🛑 CRASH ERROR DETECTED: An unhandled error occurred: {e}")
        
    print("\n------------------------------------------------")   
