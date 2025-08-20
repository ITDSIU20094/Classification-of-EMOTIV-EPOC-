import pandas as pd
import numpy as np
import glob
from pathlib import Path

# List of electrodes
electrodes = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 
    'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
]

# Attributes for each electrode
attributes = ['delta std', 'delta m', 'theta std', 'theta m', 
              'alpha std', 'alpha m', 'beta std', 'beta m']

def preprocess_dataframe(df):
    """
    Preprocess DataFrame: remove columns with all missing values
    """
    # Create a copy to avoid modifying original DataFrame
    processed_df = df.copy()
    
    # Remove columns with all missing values
    processed_df = processed_df.dropna(axis=1, how='all')
    
    # Check if Class column exists
    if 'Class' not in processed_df.columns:
        raise ValueError("'Class' column does not exist in data after missing values processing")
    
    return processed_df

def transform_dataframe(df):
    """
    Transform column names to new structure
    """
    # Create new DataFrame
    new_df = pd.DataFrame()
    new_df['Class'] = df['Class']

    # Get remaining columns (excluding Class)
    remaining_columns = [col for col in df.columns if col != 'Class']
    
    # Check if column count matches expected
    expected_electrode_columns = len(electrodes) * 8
    if len(remaining_columns) != expected_electrode_columns:
        print(f"Warning: Number of columns after processing ({len(remaining_columns)}) " 
              f"does not match expected ({expected_electrode_columns})")
        print("Continuing processing with current column count...")

    # Process each electrode
    electrode_index = 0
    col_index = 0
    
    while col_index < len(remaining_columns) and electrode_index < len(electrodes):
        electrode = electrodes[electrode_index]
        
        # Get next 8 columns for current electrode
        if col_index + 8 <= len(remaining_columns):
            # Get the actual column indices
            start_col_name = remaining_columns[col_index]
            end_col_name = remaining_columns[col_index + 7] if col_index + 7 < len(remaining_columns) else remaining_columns[-1]
            
            # Get column indices
            start_idx = df.columns.get_loc(start_col_name)
            end_idx = df.columns.get_loc(end_col_name) + 1
            
            electrode_data = df.iloc[:, start_idx:end_idx]
            
            # Ensure we have exactly 8 columns
            if len(electrode_data.columns) == 8:
                # Rename columns
                electrode_data.columns = [f"{electrode} {attr}" for attr in attributes]
                
                # Concatenate to new DataFrame
                new_df = pd.concat([new_df, electrode_data], axis=1)
                
                col_index += 8
                electrode_index += 1
            else:
                print(f"Warning: Not enough columns for electrode {electrode}")
                break
        else:
            print(f"Warning: Not enough columns for electrode {electrode}")
            break

    return new_df

def process_eeg_data(input_df, file_name=""):
    """
    Main function to process EEG data
    """
    if file_name:
        print(f"Processing file: {file_name}")
    
    # Step 1: Preprocessing - remove columns with missing values
    print("Step 1: Removing columns with missing values...")
    processed_df = preprocess_dataframe(input_df)
    print(f"Columns after missing values processing: {len(processed_df.columns)}")
    
    # Step 2: Transform column names
    print("Step 2: Transforming column names...")
    final_df = transform_dataframe(processed_df)
    print(f"Final number of columns: {len(final_df.columns)}")
    
    return final_df

def process_multiple_files(file_patterns, output_folder="processed_data"):
    """
    Process multiple CSV files simultaneously
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(exist_ok=True)
    
    results = {}
    
    # Process each file
    for file_pattern in file_patterns:
        # Find files matching the pattern
        file_paths = glob.glob(file_pattern)
        
        for file_path in file_paths:
            try:
                print(f"\n{'='*50}")
                print(f"Processing: {file_path}")
                
                # Read CSV file
                df = pd.read_csv(file_path)
                print(f"Original columns: {len(df.columns)}")
                
                # Process the data
                processed_df = process_eeg_data(df, Path(file_path).name)
                
                # Save processed file
                output_path = Path(output_folder) / f"{Path(file_path).name}"
                processed_df.to_csv(output_path, index=False)
                
                results[file_path] = {
                    'status': 'success',
                    'output_path': str(output_path),
                    'original_columns': len(df.columns),
                    'processed_columns': len(processed_df.columns)
                }
                
                print(f"Successfully processed and saved to: {output_path}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                results[file_path] = {
                    'status': 'error',
                    'error_message': str(e)
                }
    
    return results

# Example usage
if __name__ == "__main__":
    # Define your file patterns (can use wildcards)
    file_patterns = [
        "/home/quan/PROJECT/Machine Learning with Biomedical Signals/kaggle_downloads/user_a.csv",
        "/home/quan/PROJECT/Machine Learning with Biomedical Signals/kaggle_downloads/user_b.csv", 
        "/home/quan/PROJECT/Machine Learning with Biomedical Signals/kaggle_downloads/user_c.csv",
        "/home/quan/PROJECT/Machine Learning with Biomedical Signals/kaggle_downloads/user_d.csv"
    ]
    
    # Process all files
    processing_results = process_multiple_files(file_patterns, output_folder="cleaned_data")
    
    # Print summary
    print(f"\n{'='*50}")
    print("PROCESSING SUMMARY:")
    print(f"{'='*50}")
    
    for file_path, result in processing_results.items():
        if result['status'] == 'success':
            print(f"{Path(file_path).name}: SUCCESS "
                  f"({result['original_columns']} → {result['processed_columns']} columns)")
        else:
            print(f"{Path(file_path).name}: ERROR - {result['error_message']}")