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

def clean_and_validate_dask(ddf):
    """
    Clean data - Dask version with statistics
    """
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

def quantile_normalization_dask(ddf, n_bins=10):
    """
    Quantile normalization - PRESERVING MISSING VALUES
    Only normalize non-missing values, keep NaN as NaN
    """
    print(f"Normalizing with {n_bins} bins (preserving missing values)...")
    
    def normalize_partition(df):
        """Normalize each partition while preserving missing values"""
        if len(df) == 0:
            return df
        
        def normalize_group(group):
            """Normalize each loinc_code group, preserving missing values"""
            # Separate missing and non-missing values
            mask_missing = group.isna()
            non_missing = group[~mask_missing]
            
            # If all values are missing, return as is
            if len(non_missing) == 0:
                return group
            
            log_transformed = np.log1p(non_missing)
            n_unique = log_transformed.nunique()
            
            # Normalize only non-missing values
            try:
                normalized = pd.qcut(log_transformed, q=n_bins, labels=False, duplicates='drop')
            except ValueError:
                if n_unique > 1:
                    try:
                        normalized = pd.cut(log_transformed, bins=min(n_unique, n_bins), 
                                          labels=False, include_lowest=True)
                    except:
                        normalized = pd.Series(0, index=log_transformed.index)
                else:
                    normalized = pd.Series(0, index=log_transformed.index)
            
            # Reconstruct full series: keep NaN in their original positions
            result = pd.Series(index=group.index, dtype='float64')
            result[~mask_missing] = normalized.astype('float64')
            result[mask_missing] = np.nan  # Preserve NaN
            
            return result
        
        # Normalize each loinc_code within this partition
        df = df.copy()
        df['value'] = df.groupby('loinc_code')['value'].transform(normalize_group)
        return df
    
    # Apply to all partitions
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
    print("(Preserving Missing Values)")
    print("="*60)
    
    print("\nStep 1: Loading data")
    print("-"*60)
    ddf = load_and_filter_data_dask(input_file)
    
    print("\nStep 2: Cleaning and validating")
    print("-"*60)
    ddf_clean = clean_and_validate_dask(ddf)
    
    print("\nStep 3: Quantile normalization (preserving missing values)")
    print("-"*60)
    ddf_normalized = quantile_normalization_dask(ddf_clean, n_bins=10)
    
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