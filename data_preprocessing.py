import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
from dask.diagnostics import ProgressBar

def load_and_filter_data_dask(file_path):
    """
    Load large file with Dask, automatic chunking and parallel processing
    """
    print("Loading data with Dask (parallel)...")
    
    dtypes = {
        'subject_id': 'int32',
        'valuenum': 'float32',
        'omop_concept_id': 'str'
    }
    
    # Dask automatically splits file into chunks, 256MB each
    ddf = dd.read_csv(
        file_path,
        usecols=['subject_id', 'charttime', 'valuenum', 'omop_concept_id'],
        dtype=dtypes,
        blocksize='256MB', 
        assume_missing=True
    )
    
    # Rename columns
    ddf = ddf.rename(columns={
        'subject_id': 'patient_id',
        'charttime': 'charttime',
        'valuenum': 'value',
        'omop_concept_id': 'loinc_code'
    })
    
    print(f"Data loaded into {ddf.npartitions} partitions")
    return ddf

def transform(x):
    return np.sign(x) * np.log(np.abs(x) + 1)

def clean_and_validate_dask(ddf):
    print("Cleaning and validating data...")
    
    # Count initial rows
    print("Counting total records...")
    initial_count = len(ddf)
    print(f"Total records: {initial_count:,}")
    
    # Remove rows without loinc_code
    ddf = ddf.dropna(subset=['loinc_code'])
    after_drop_na = len(ddf)
    removed_no_loinc = initial_count - after_drop_na
    pct_no_loinc = (removed_no_loinc / initial_count * 100) if initial_count > 0 else 0
    print(f"Without loinc code: {removed_no_loinc:,} (around {pct_no_loinc:.2f}%)")
    
    # Data type conversion
    ddf['patient_id'] = ddf['patient_id'].astype(str)
    ddf['charttime'] = dd.to_datetime(ddf['charttime'], errors='coerce')
    ddf['value'] = dd.to_numeric(ddf['value'], errors='coerce')
    ddf['loinc_code'] = ddf['loinc_code'].astype(str)
    
    # Remove duplicates
    print("Removing duplicates...")
    before_dedup = len(ddf)
    ddf = ddf.drop_duplicates(
        subset=['patient_id', 'loinc_code', 'charttime', 'value']
    )
    after_dedup = len(ddf)
    removed_duplicates = before_dedup - after_dedup
    pct_duplicates = (removed_duplicates / initial_count * 100) if initial_count > 0 else 0
    print(f"Duplicate records: {removed_duplicates:,} (around {pct_duplicates:.2f}%)")
    
    # Final statistics
    print(f"Cleaned data: {after_dedup:,}")
    
    # Count rows with/without values
    print("Counting rows with values...")
    rows_with_values = ddf['value'].notnull().sum().compute()
    rows_missing_values = after_dedup - rows_with_values
    print(f"  Rows with values: {rows_with_values:,}")
    print(f"  Rows with missing values: {rows_missing_values:,}")
    
    return ddf

def minmax_normalization(ddf, n_bins=10):
    print(f"Computing global min/max for each loinc_code...")

    ddf_valid = ddf[ddf['value'].notnull()]
    global_stats = ddf_valid.groupby('loinc_code')['value'].agg(['min', 'max']).compute()
    global_stats = global_stats.reset_index()
    global_stats.columns = ['loinc_code', 'MIN_VALUE', 'MAX_VALUE']

    global_stats = global_stats[global_stats['MIN_VALUE'] != global_stats['MAX_VALUE']]
    print(f"Normalizing with {n_bins} equal-width bins for {len(global_stats)} loinc_codes...")

    def normalize_partition(df):
        if len(df) == 0:
            return df
        
        df = df.merge(global_stats, how='left', on='loinc_code')
        has_stats = df['MIN_VALUE'].notna() & df['MAX_VALUE'].notna()
        has_value = df['value'].notna()
        valid_mask = has_stats & has_value

        if valid_mask.sum() > 0:
            valid_values = df.loc[valid_mask, 'value'].astype(float)
            valid_mins = df.loc[valid_mask, 'MIN_VALUE'].astype(float)
            valid_maxs = df.loc[valid_mask, 'MAX_VALUE'].astype(float)
        
            t_val = transform(valid_values)
            t_min = transform(valid_mins)
            t_max = transform(valid_maxs)

            # Normalize
            span = t_max - t_min
            pos = ((t_val - t_min) / span).clip(lower=0, upper=1 - 1e-12)
            bins = np.floor(pos * n_bins).astype(float)

            df.loc[valid_mask, 'value'] = bins.astype(float)
        
        df.loc[~valid_mask, 'value'] = np.nan

        df = df.drop(columns=['MIN_VALUE', 'MAX_VALUE'], errors='ignore')

        return df
    ddf = ddf.map_partitions(normalize_partition, meta=ddf)
    return ddf  

def main():
    """
    Main function
    """
    input_file = 'data/d_labevent_with_loinc.csv'
    output_path = 'data/clean/norm'
    
    print("="*60)
    print("DASK Parallel Processing Pipeline")
    print("(Min-Max Normalization - Method 1)")
    print("="*60)
    
    print("\nStep 1: Loading data")
    print("-"*60)
    ddf = load_and_filter_data_dask(input_file)
    
    print("\nStep 2: Cleaning and validating")
    print("-"*60)
    ddf_clean = clean_and_validate_dask(ddf)
    
    print("\nStep 3: Min-max normalization (sign-preserving log + linear)")
    print("-"*60)
    ddf_normalized = minmax_normalization(ddf_clean, n_bins=10)
    
    # Check missing values after normalization
    print("\nPost-normalization statistics:")
    print("-"*60)
    total_rows = len(ddf_normalized)
    rows_with_values = ddf_normalized['value'].notnull().sum().compute()
    rows_missing_values = total_rows - rows_with_values
    pct_missing = (rows_missing_values / total_rows * 100) if total_rows > 0 else 0
    print(f"Total rows: {total_rows:,}")
    print(f"  Rows with normalized values: {rows_with_values:,}")
    print(f"  Rows with missing values: {rows_missing_values:,} ({pct_missing:.2f}%)")
    
    print("\nStep 4: Computing and saving results")
    print("-"*60)
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'normalized_data.csv')
    
    print("Computing final results...")
    ddf_normalized.to_csv(
        output_file,
        single_file=True,
        index=False
    )
    
    # Final statistics
    print("\n" + "="*60)
    print("Processing completed successfully!")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()