import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split 

"""
This file is used for extract target columns from MIMIC-IV dataset with loinc code mapping. And also normalized item values using ref_range.
Processed dataset are saved in ./data/raw/ and ./data/clean/:
Raw extracted dataset (with cleaning duplicate rows and missing reference range & item value row): ./data/raw/clean_data.csv
Normalized dataset: ./data/clean/norm/normalized_data.csv
Split into train & test set from normalized dataset: ./data/clean/train_data.csv, ./data/clean/test_data.csv
"""

def load_and_filter_data(file_path, output_path):
    """
    Load data and rename columns:
    original: [subject_id, itemid, charttime, valuenum, valueuom_y, ref_range_lower,ref_range_upper, omop_concept_id, omop_concept_name] 
    as: [patient_id, itemid, charttime, value, unit, ref_range_lower,ref_range_upper, loinc_code, item_name] 
    """
    df = pd.read_csv(file_path)
    required_columns = {
        'subject_id': 'patient_id',
        'itemid': 'itemid', 
        'charttime': 'charttime',
        'valuenum': 'value',
        'valueuom_y': 'unit', # Using loinc code unit
        'ref_range_lower': 'ref_range_lower',
        'ref_range_upper': 'ref_range_upper',
        'omop_concept_id': 'loinc_code',
        'omop_concept_name': 'item_name'
    }
    df_filtered = df[list(required_columns.keys())].copy()
    df_filtered = df_filtered.rename(columns=required_columns)
    return df_filtered

def clean_and_validate(df):
    """
    Remove rows with missing loinc code.
    """
    df_clean = df.copy()
    # Only remove rows without loinc_code (essential identifier)
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['loinc_code'])
    removed_no_loinc = initial_count - len(df_clean)
    if removed_no_loinc > 0:
        print(f"Removed {removed_no_loinc} rows without loinc_code")
    # data type convertion
    df_clean['patient_id'] = df_clean['patient_id'].astype(str)
    df_clean['itemid'] = df_clean['itemid'].astype(str)
    df_clean['charttime'] = pd.to_datetime(df_clean['charttime'])
    df_clean['value'] = pd.to_numeric(df_clean['value'], errors='coerce')
    df_clean['unit'] = df_clean['unit'].astype(str)
    df_clean['ref_range_lower'] = pd.to_numeric(df_clean['ref_range_lower'], errors='coerce')
    df_clean['ref_range_upper'] = pd.to_numeric(df_clean['ref_range_upper'], errors='coerce')
    df_clean['loinc_code'] = df_clean['loinc_code'].astype(str)
    df_clean['item_name'] = df_clean['item_name'].astype(str)

    # remove invalid range value
    mask_invalid_range = (
        df_clean['ref_range_lower'].notna() & 
        df_clean['ref_range_upper'].notna() & 
        (df_clean['ref_range_lower'] > df_clean['ref_range_upper'])
    )
    invalid_count = mask_invalid_range.sum()
    if invalid_count > 0:
        # Marking these ranges as NaN for later lookup
        print(f"\nFound {invalid_count} rows with invalid ranges (lower > upper)")
        # Set invalid ranges to NaN - will be handled same as missing ranges
        df_clean.loc[mask_invalid_range, 'ref_range_lower'] = np.nan
        df_clean.loc[mask_invalid_range, 'ref_range_upper'] = np.nan
    return df_clean

def normalization(df_clean, output_path):
    """
    Normalize the values in the dataframe using min-max normalization with reference range.
    Steps:
    1. Build reference range mapping for each itemid
    2. Fill missing ranges using the mapping
    3. Normalize values where ranges are available
    4. Keep all rows, but only remove rows with value but no range after lookup
    """
    # Build reference range mapping for each itemid
    print("\n" + "="*60)
    print("Step 1: Building reference range mapping for each itemid")
    print("="*60)
    valid_range_mask = (
        df_clean['ref_range_lower'].notna() &
        df_clean['ref_range_upper'].notna()
    )
    df_with_valid_range = df_clean[valid_range_mask].copy()
    # Build range mapping for those items with value but lack range reference
    range_mapping = {}
    conflicting_items = {}
    for itemid in df_with_valid_range['itemid'].unique():
        item_data = df_with_valid_range[df_with_valid_range['itemid'] == itemid]
        unique_ranges = item_data[['ref_range_lower', 'ref_range_upper']].drop_duplicates()
        if len(unique_ranges) == 1:
            range_mapping[itemid] = {
                'ref_range_lower': unique_ranges.iloc[0]['ref_range_lower'],
                'ref_range_upper': unique_ranges.iloc[0]['ref_range_upper'],
                'unit': item_data['unit'].mode()[0] if len(item_data['unit'].mode()) > 0 else item_data['unit'].iloc[0],
                'loinc_code': item_data['loinc_code'].iloc[0],
                'item_name': item_data['item_name'].iloc[0],
                'occurrences': len(item_data)
            }
        else:
            conflicting_items[itemid] = {
                'item_name': item_data['item_name'].iloc[0],
                'loinc_code': item_data['loinc_code'].iloc[0],
                'num_different_ranges': len(unique_ranges),
                'ranges': unique_ranges.values.tolist(),
                'occurrences': len(item_data)
            }
    print(f"Found {len(range_mapping)} items with consistent reference ranges")
    print(f"Found {len(conflicting_items)} itemids with conflicting ranges (will NOT fill)")
    # Print conflicting items details
    if conflicting_items:
        print("\n" + "="*60)
        print("WARNING: Items with conflicting reference ranges")
        print("="*60)
        print("These itemids have multiple different reference ranges.")
        print("Missing ranges for these items will NOT be filled.\n")
        
        for itemid, info in conflicting_items.items():
            print(f"ItemID: {itemid}")
            print(f"  Name: {info['item_name']}")
            print(f"  LOINC: {info['loinc_code']}")
            print(f"  Total occurrences: {info['occurrences']}")
            print(f"  Number of different ranges: {info['num_different_ranges']}")
            print(f"  Ranges found:")
            for lower, upper in info['ranges']:
                print(f"    Lower: {lower}, Upper: {upper}")
            print()

    # Step 2: Fill missing ranges using the mapping (only for consistent items)
    df_processed = df_clean.copy()
    
    # Identify rows needing range lookup
    need_range_mask = (
        df_processed['ref_range_lower'].isna() | 
        df_processed['ref_range_upper'].isna()
    )
    filled_count = 0
    skipped_conflicting = 0
    for idx in df_processed[need_range_mask].index:
        itemid = df_processed.loc[idx, 'itemid']
        
        if itemid in range_mapping:
            # Only fill if itemid has consistent ranges
            df_processed.loc[idx, 'ref_range_lower'] = range_mapping[itemid]['ref_range_lower']
            df_processed.loc[idx, 'ref_range_upper'] = range_mapping[itemid]['ref_range_upper']
            filled_count += 1
        elif itemid in conflicting_items:
            # Skip filling for items with conflicting ranges
            skipped_conflicting += 1

    # Step 3: Categorize all rows
    has_value = df_processed['value'].notna()
    has_complete_range = (
        df_processed['ref_range_lower'].notna() & 
        df_processed['ref_range_upper'].notna()
    )
    
    # Category 1: Has value and complete range -> normalize
    mask_normalize = has_value & has_complete_range
    
    # Category 2: Has value but no range even after lookup -> REMOVE
    mask_remove = has_value & ~has_complete_range
    
    # Category 3: No value -> keep as is
    mask_no_value = ~has_value
    print(f"Category 1 - Will normalize (has value & range): {mask_normalize.sum()}")
    print(f"Category 2 - Will REMOVE (has value but no range): {mask_remove.sum()}")
    print(f"Category 3 - Will keep as-is (no value): {mask_no_value.sum()}")
    # Print details of rows that will be removed
    if mask_remove.sum() > 0:
        print("\n" + "="*60)
        print("ROWS TO BE REMOVED (has value but no range after lookup):")
        print("="*60)
        df_to_remove = df_processed[mask_remove][['patient_id', 'itemid', 'value', 'loinc_code', 'item_name']].copy()
        
        # Group by itemid to show summary
        print("\nSummary by itemid:")
        removal_summary = df_to_remove.groupby(['itemid', 'loinc_code', 'item_name']).size().reset_index(name='count')
        removal_summary = removal_summary.sort_values('count', ascending=False)
        print(removal_summary.to_string(index=False))
        
        print(f"\nTotal unique itemids affected: {df_to_remove['itemid'].nunique()}")
        print(f"Total rows to be removed: {len(df_to_remove)}")
        
        # Show which removed items had conflicting ranges
        removed_itemids = set(df_to_remove['itemid'].unique())
        conflicting_removed = removed_itemids.intersection(set(conflicting_items.keys()))
        if conflicting_removed:
            print(f"\nOf these, {len(conflicting_removed)} itemids were removed due to conflicting ranges:")
            for itemid in list(conflicting_removed)[:5]:
                print(f"  ItemID {itemid}: {conflicting_items[itemid]['item_name']}")
            if len(conflicting_removed) > 5:
                print(f"  ... and {len(conflicting_removed) - 5} more")

    df_to_normalize = df_processed[mask_normalize].copy()
    
    def calculate_min_max_norm(row):
        value = row['value']
        min_val = row['ref_range_lower'] 
        max_val = row['ref_range_upper']  

        if pd.isna(value):
            return value

        range_diff = max_val - min_val
        
        if range_diff == 0:
            return 0.5
        
        normalized_value = (value - min_val) / range_diff
        return normalized_value

    df_to_normalize['value'] = df_to_normalize.apply(calculate_min_max_norm, axis=1)
    normalized_values = df_to_normalize['value'].dropna()
    if len(normalized_values) > 0:
        print(f"Min-Max normalization statistics:")
        print(f"  Total normalized values: {len(normalized_values)}")
        print(f"  Min normalized value: {normalized_values.min():.3f}")
        print(f"  Max normalized value: {normalized_values.max():.3f}")
        print(f"  Mean normalized value: {normalized_values.mean():.3f}")
    
    # Step 5: Combine Category 1 and Category 3, exclude Category 2
    print("\n" + "="*60)
    print("Step 5: Creating final dataset")
    print("="*60)
    
    df_keep_no_value = df_processed[mask_no_value].copy()
    df_final = pd.concat([df_to_normalize, df_keep_no_value], ignore_index=True)
    
    # Sort by patient_id and charttime
    df_final = df_final.sort_values(['patient_id', 'charttime']).reset_index(drop=True)
    
    # Select only target columns
    target_columns = ['patient_id', 'charttime', 'itemid', 'value', 'unit', 'loinc_code']
    df_output = df_final[target_columns].copy()
    
    print(f"\nFinal dataset statistics:")
    print(f"  Total rows kept: {len(df_output)}")
    print(f"  Rows with normalized values: {mask_normalize.sum()}")
    print(f"  Rows without values (kept): {mask_no_value.sum()}")
    print(f"  Rows removed: {mask_remove.sum()}")
    
    # Save to output file
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'normalized_data.csv')
    df_output.to_csv(output_file, index=False)
    print(f"\nSaved normalized data to: {output_file}")

    return df_output

def split(df, test_size=0.2, random_state=42, split_by='patient'):
    """
    Spilt datasets into training set & testing set.
    """
    if split_by == 'patient':
        unique_patients = df['patient_id'].unique()
        train_patients, test_patients = train_test_split(
            unique_patients,
            test_size=test_size,
            random_state=random_state
        )
        train_df = df[df['patient_id'].isin(train_patients)].copy()
        test_df = df[df['patient_id'].isin(test_patients)].copy()
    elif split_by == 'random':
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state
        )
        # Check for patient overlap (for information)
        train_patients = set(train_df['patient_id'].unique())
        test_patients = set(test_df['patient_id'].unique())
        overlap = len(train_patients & test_patients)
        if overlap > 0:
            print(f"  Warning: {overlap} patients appear in both train and test sets")
        
    else:
        raise ValueError("split_by must be either 'patient' or 'random'")
    
    return train_df, test_df

def save_split_data(train_df, test_df, train_path, test_path):
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Save files
    train_file = os.path.join(train_path, 'train_data.csv')
    test_file = os.path.join(test_path, 'test_data.csv')
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    return train_file, test_file

def main():
    """
    Main function to process the lab data through the complete pipeline.
    """
    input_file = 'filtered_lab_with_loinc.csv'
    base_output_path = 'data'
    raw_output_path = os.path.join(base_output_path, 'raw')
    clean_norm_path = os.path.join(base_output_path, 'clean', 'norm')
    clean_train_path = os.path.join(base_output_path, 'clean', 'train')
    clean_test_path = os.path.join(base_output_path, 'clean', 'test')

    df_filtered = load_and_filter_data(input_file, raw_output_path)

    os.makedirs(raw_output_path, exist_ok=True)
    filtered_output_file = os.path.join(raw_output_path, 'filtered_data.csv')
    df_filtered.to_csv(filtered_output_file, index=False)

    df_clean = clean_and_validate(df_filtered)

    clean_output_file = os.path.join(raw_output_path, 'clean_data.csv')
    df_clean.to_csv(clean_output_file, index=False)

    df_normalized = normalization(df_clean, clean_norm_path)

    train_df, test_df = split(
        df_normalized, 
        test_size=0.2, 
        random_state=42, 
        split_by='patient'
    )
    save_split_data(train_df, test_df, clean_train_path, clean_test_path)

    return df_filtered, df_clean, df_normalized, train_df, test_df

if __name__ == "__main__":
    df_filtered, df_clean, df_normalized, train_df, test_df = main()
