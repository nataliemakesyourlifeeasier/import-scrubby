# import-scrubby
Streamlit app to scrub and validate csv Reservation Import files

1) Accepts an excel file upload.
2) Scrubs the data and defaults certain values
     - ageCategories defaulted to 0 if at least 1 ageCategory non-0/not null for row
     - pets defaulted to 0 if null
4) Validates data
5) If Validation fails, will give report with specific rows and data values needed to proceed
6) If Validation passes, creates a downloadable .csv file to upload to ReservationImport task
