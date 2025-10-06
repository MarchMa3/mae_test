import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

"""
This file is used for extract target columns from MIMIC-IV dataset with loinc code mapping.
Processed dataset are saved in ./data/raw/ and ./data/clean/
Raw extracted dataset (with cleaning duplicate rows and missing loinc code): ./data/raw/clean_data.csv
Split into train & test set: ./data/clean/train_data.csv, ./data/clean/test_data.csv
"""

def load_and_filter_data(file_path):
    """
    Load data and rename columns:
    original: [subject_id, charttime, valuenum, omop_concept_id] 
    as: [patient_id, charttime, value, loinc_code] 
    """
    df = pd.read_csv(file_path)
    required_columns = {
        'subject_id': 'patient_id',
        'charttime': 'charttime',
        'valuenum': 'value',
        'omop_concept_id': 'loinc_code'
    }
    df_filtered = df[list(required_columns.keys())].copy()
    df_filtered = df_filtered.rename(columns=required_columns)
    return df_filtered

def clean_and_validate(df):
    """
    Remove rows with missing loinc code & duplicated values.
    """
    df_clean = df.copy()
    
    # Remove rows without loinc_code
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['loinc_code'])
    removed_no_loinc = initial_count - len(df_clean)
    if removed_no_loinc > 0:
        print(f"Removed {removed_no_loinc} rows without loinc_code")
    
    # Data type conversion
    df_clean['patient_id'] = df_clean['patient_id'].astype(str)
    df_clean['charttime'] = pd.to_datetime(df_clean['charttime'])
    df_clean['value'] = pd.to_numeric(df_clean['value'], errors='coerce')
    df_clean['loinc_code'] = df_clean['loinc_code'].astype(str)

    # Remove duplicated values
    before_dedup = len(df_clean)
    duplicate_mask = df_clean.duplicated(
        subset=['patient_id', 'loinc_code', 'charttime', 'value'], 
        keep='first'
    )
    
    df_clean = df_clean[~duplicate_mask].copy()
    after_dedup = len(df_clean)
    
    removed = before_dedup - after_dedup
    if removed > 0:
        print(f"Removed {removed} duplicate records")
    
    return df_clean

def quantile_normalization(df, n_bins=10, output_path=None):
    """
    Quantile normalization for each lab test (loinc code).

    Args:
        df: Dataframe
        n_bins: number of bins
        output_path: path to save normalized data
    """
    df_normalized = df.copy()
    df_normalized['original_value'] = df_normalized['value'].copy()
    df_normalized['normalized_value'] = np.nan 

    unique_loinc_codes = df['loinc_code'].unique()
    print(f"Normalizing {len(unique_loinc_codes)} unique lab tests with {n_bins} bins")

    loinc_stats = {}

    # Process each lab test
    for loinc_code in unique_loinc_codes:
        mask = (df_normalized['loinc_code'] == loinc_code) & (df_normalized['original_value'].notna())
        if mask.sum() == 0:
            continue
        
        values = df_normalized.loc[mask, 'original_value']
        n_unique = values.nunique()
        
        # Calculate quantile bins
        try:
            normalized_bins, bin_edges = pd.qcut(
                values,
                q=n_bins,
                labels=False,
                retbins=True,
                duplicates='drop'
            )
            df_normalized.loc[mask, 'normalized_value'] = normalized_bins
            loinc_stats[loinc_code] = {
                'n_samples': len(values),
                'n_unique_values': n_unique,
                'n_bins_created': len(bin_edges) - 1,
                'min_value': values.min(),
                'max_value': values.max(),
                'bin_edges': bin_edges.tolist()
            }
        except ValueError:
            # Fallback for edge cases
            if n_unique > 1:
                try:
                    normalized_bins = pd.cut(values, bins=n_unique, labels=False, include_lowest=True)
                    df_normalized.loc[mask, 'normalized_value'] = normalized_bins
                    loinc_stats[loinc_code] = {
                        'n_samples': len(values),
                        'n_unique_values': n_unique,
                        'n_bins_created': n_unique,
                        'min_value': values.min(),
                        'max_value': values.max()
                    }
                except Exception:
                    pass
            elif n_unique == 1:
                df_normalized.loc[mask, 'normalized_value'] = 0
                loinc_stats[loinc_code] = {
                    'n_samples': len(values),
                    'n_unique_values': 1,
                    'n_bins_created': 1,
                    'min_value': values.min(),
                    'max_value': values.max()
                }
    
    # Replace original value with normalized value
    df_normalized['value'] = df_normalized['normalized_value']
    
    # Summary
    rows_normalized = df_normalized['value'].notna().sum()
    print(f"Successfully normalized {rows_normalized} rows")
    
    # Drop temporary columns
    df_output = df_normalized.drop(columns=['original_value', 'normalized_value'])
    
    # Save if output path provided
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        
        # Save normalized data
        output_file = os.path.join(output_path, 'normalized_data.csv')
        output_columns = ['patient_id', 'charttime', 'value', 'loinc_code']
        df_output[output_columns].to_csv(output_file, index=False)
        print(f"Saved normalized data to: {output_file}")
        
        # Save statistics
        stats_file = os.path.join(output_path, 'normalization_stats.csv')
        stats_records = [
            {
                'loinc_code': loinc,
                'n_samples': stats['n_samples'],
                'n_unique_values': stats['n_unique_values'],
                'n_bins_created': stats['n_bins_created'],
                'original_min_value': stats['min_value'],
                'original_max_value': stats['max_value']
            }
            for loinc, stats in loinc_stats.items()
        ]
        pd.DataFrame(stats_records).to_csv(stats_file, index=False)
        print(f"Saved statistics to: {stats_file}")
    
    return df_output
    

def main():
    """
    Main function to process the lab data through the complete pipeline.
    """
    input_file = 'data/sample_lab_with_loinc.csv'
    base_output_path = 'data'
    clean_norm_path = os.path.join(base_output_path, 'clean', 'norm')
    clean_train_path = os.path.join(base_output_path, 'clean', 'train')
    clean_test_path = os.path.join(base_output_path, 'clean', 'test')

    print("="*60)
    print("Step 1: Loading and filtering data")
    print("="*60)
    df_filtered = load_and_filter_data(input_file)
    print(f"Loaded {len(df_filtered)} rows with columns: {list(df_filtered.columns)}")

    print("\n" + "="*60)
    print("Step 2: Cleaning and validating data")
    print("="*60)
    df_clean = clean_and_validate(df_filtered)
    print(f"\nCleaned data: {len(df_clean)} rows")
    print(f"  Rows with values: {df_clean['value'].notna().sum()}")
    print(f"  Rows with missing values: {df_clean['value'].isna().sum()}")

    print("\n" + "="*60)
    print("Step 3: Quantile normalization")
    print("="*60)
    print("Note: The 'value' column will be replaced with normalized bin numbers (0-19)")
    df_normalized = quantile_normalization(df_clean, n_bins=10, output_path=clean_norm_path)

    print("\n" + "="*60)
    print("Processing completed successfully!")
    print("="*60)
    print(f"Final dataset contains columns: {list(df_normalized.columns)}")
    if df_normalized['value'].notna().sum() > 0:
        print(f"The 'value' column now contains normalized bin numbers (0-{df_normalized['value'].max():.0f})")

    return df_filtered, df_clean, df_normalized, train_df, test_df

if __name__ == "__main__":
    df_filtered, df_clean, df_normalized, train_df, test_df = main()
